import collections
import warnings
from copy import deepcopy
from functools import partial
from itertools import product
from json import loads
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import skimage.io
import xarray as xr
from skimage import img_as_float32
from slicedimage import (
    ImageFormat,
    Reader,
    Tile,
    TileSet,
    Writer,
)
from slicedimage.io import resolve_path_or_url
from tqdm import tqdm

from starfish.core.config import StarfishConfig
from starfish.core.errors import DataFormatWarning
from starfish.core.imagestack import indexing_utils, physical_coordinate_calculator
from starfish.core.imagestack.parser import TileCollectionData, TileKey
from starfish.core.imagestack.parser.crop import CropParameters, CroppedTileCollectionData
from starfish.core.imagestack.parser.numpy import NumpyData
from starfish.core.imagestack.parser.tileset import TileSetData
from starfish.core.multiprocessing.pool import Pool
from starfish.core.multiprocessing.shmem import SharedMemory
from starfish.core.types import (
    Axes,
    Clip,
    Coordinates,
    LOG,
    Number,
    STARFISH_EXTRAS_KEY
)
from starfish.core.util import logging
from starfish.core.util.dtype import preserve_float_range
from ._mp_dataarray import MPDataArray
from .dataorder import AXES_DATA, N_AXES


class ImageStack:
    """
    ImageStacks are the main objects for processing images in starfish. It is a
    5-dimensional Image Tensor that labels each (z, y, x) tile with the round and channel
    and zplane it corresponds to. The class is a wrapper around :py:class:`xarray.DataArray`.
    The names of each imaging r/ch/zplane as well as the physical coordinates of each tile
    are stored as coordinates on the :py:class:`xarray.DataArray`. ImageStacks can only be
    initialized with aligned Tilesets.


    Loads configuration from StarfishConfig.

    Attributes
    ----------
    num_chs : int
        the number of channels stored in the image tensor
    num_rounds : int
        the number of imaging rounds stored in the image tensor
    num_zplanes : int
        the number of z-layers stored in the image tensor
    xarray : :py:class:`xarray.DataArray`
        the 5-d image tensor is stored in this array
    raw_shape : Tuple[int]
        the shape of the image tensor (in integers)
    shape : Dict[str, int]
           the shape of the image tensor by categorical index (channels, imaging rounds, z-layers)
    """

    def __init__(
            self,
            tile_data: TileCollectionData,
    ) -> None:
        axes_sizes = {
            Axes.ROUND: len(set(tilekey.round for tilekey in tile_data.keys())),
            Axes.CH: len(set(tilekey.ch for tilekey in tile_data.keys())),
            Axes.ZPLANE: len(set(tilekey.z for tilekey in tile_data.keys())),
        }

        self._tile_data = tile_data

        # check for existing log info
        if STARFISH_EXTRAS_KEY in tile_data.extras and LOG in tile_data.extras[STARFISH_EXTRAS_KEY]:
            self._log = loads(tile_data.extras[STARFISH_EXTRAS_KEY])[LOG]
        else:
            self._log: List[dict] = list()

        data_shape: MutableSequence[int] = []
        data_dimensions: MutableSequence[str] = []
        data_tick_marks: MutableMapping[str, Sequence[int]] = dict()
        for ix in range(N_AXES):
            size_for_axis: Optional[int] = None
            dim_for_axis: Optional[Axes] = None

            for axis_name, axis_data in AXES_DATA.items():
                if ix == axis_data.order:
                    size_for_axis = axes_sizes[axis_name]
                    dim_for_axis = axis_name
                    break

            if size_for_axis is None or dim_for_axis is None:
                raise ValueError(
                    f"Could not find entry for the {ix}th axis in AXES_DATA")

            data_shape.append(size_for_axis)
            data_dimensions.append(dim_for_axis.value)
            data_tick_marks[dim_for_axis.value] = list(
                sorted(set(tilekey[dim_for_axis] for tilekey in self._tile_data.keys())))

        data_shape.extend([tile_data.tile_shape[Axes.Y], tile_data.tile_shape[Axes.X]])
        data_dimensions.extend([Axes.Y.value, Axes.X.value])

        # now that we know the tile data type (kind and size), we can allocate the data array.
        self._data = MPDataArray.from_shape_and_dtype(
            shape=data_shape,
            dtype=np.float32,
            initial_value=0,
            dims=data_dimensions,
            coords=data_tick_marks,
        )

        all_selectors = list(self._iter_axes({Axes.ROUND, Axes.CH, Axes.ZPLANE}))
        first_selector = all_selectors[0]
        tile = tile_data.get_tile(r=first_selector[Axes.ROUND],
                                  ch=first_selector[Axes.CH],
                                  z=first_selector[Axes.ZPLANE])

        # Set up coordinates
        self._data[Coordinates.X.value] = xr.DataArray(
            np.linspace(tile.coordinates[Coordinates.X][0], tile.coordinates[Coordinates.X][1],
                        self.xarray.sizes[Axes.X.value]), dims=Axes.X.value)
        self._data[Coordinates.Y.value] = xr.DataArray(
            np.linspace(tile.coordinates[Coordinates.Y][0], tile.coordinates[Coordinates.Y][1],
                        self.xarray.sizes[Axes.Y.value]), dims=Axes.Y.value)
        if Coordinates.Z in tile.coordinates:
            # Fill with zeros for now, then replace with calculated midpoints
            self._data[Coordinates.Z.value] = xr.DataArray(np.zeros(
                self.xarray.sizes[Axes.ZPLANE.value]),
                dims=Axes.ZPLANE.value)

        # Get coords on first tile, then verify all subsequent tiles are aligned
        starting_coords = [
            tile.coordinates[Coordinates.X][0], tile.coordinates[Coordinates.X][1],
            tile.coordinates[Coordinates.Y][0], tile.coordinates[Coordinates.Y][1],
        ]

        tile_dtypes = set()
        for selector in tqdm(all_selectors):
            tile = tile_data.get_tile(
                r=selector[Axes.ROUND], ch=selector[Axes.CH], z=selector[Axes.ZPLANE])
            data = tile.numpy_array
            tile_dtypes.add(data.dtype)

            data = img_as_float32(data)
            self.set_slice(selector=selector, data=data)

            coordinates_values = [
                tile.coordinates[Coordinates.X][0], tile.coordinates[Coordinates.X][1],
                tile.coordinates[Coordinates.Y][0], tile.coordinates[Coordinates.Y][1],
            ]
            if starting_coords != coordinates_values:
                raise ValueError(
                    f"Tiles must be aligned")
            if Coordinates.Z in tile.coordinates:
                z_range = (tile.coordinates[Coordinates.Z][0], tile.coordinates[Coordinates.Z][1])
                # Use mid-point of the z range for a tile for the z-coordinate
                self._data[Coordinates.Z.value].loc[selector[Axes.ZPLANE]] = \
                    physical_coordinate_calculator.get_physical_coordinates_of_z_plane(z_range)

        tile_dtype_kinds = set(tile_dtype.kind for tile_dtype in tile_dtypes)
        tile_dtype_sizes = set(tile_dtype.itemsize for tile_dtype in tile_dtypes)
        if len(tile_dtype_kinds) != 1:
            raise TypeError("All tiles should have the same kind of dtype")
        if len(tile_dtype_sizes) != 1:
            warnings.warn("Not all tiles have the same precision data", DataFormatWarning)

    @staticmethod
    def _validate_data_dtype_and_range(data: Union[np.ndarray, xr.DataArray]) -> None:
        """verify that data is of dtype float32 and in range [0, 1]"""
        if data.dtype != np.float32:
            raise TypeError(
                f"ImageStack data must be of type float32, not {data.dtype}. Please convert data "
                f"using skimage.img_as_float32 prior to calling set_slice."
            )
        if np.min(data) < 0 or np.max(data) > 1:
            raise ValueError(
                f"ImageStack data must be of type float32 and in the range [0, 1]. Please convert "
                f"data using skimage.img_as_float32 prior to calling set_slice."
            )

    def __repr__(self):
        shape = ', '.join(f'{k}: {v}' for k, v in self._data.sizes.items())
        return f"<starfish.ImageStack ({shape})>"

    @classmethod
    def from_tileset(
            cls,
            tileset: TileSet,
            crop_parameters: Optional[CropParameters]=None,
    ) -> "ImageStack":
        """
        Parse a :py:class:`slicedimage.TileSet` into an ImageStack.

        Parameters
        ----------
        tileset : TileSet
            The tileset to parse.
        crop_parameters : Optional[CropParameters]


        Returns
        -------
        ImageStack :
            An ImageStack representing encapsulating the data from the TileSet.
        """
        tile_data: TileCollectionData = TileSetData(tileset)
        if crop_parameters is not None:
            tile_data = CroppedTileCollectionData(tile_data, crop_parameters)
        return cls(tile_data)

    @classmethod
    def from_url(cls, url: str, baseurl: Optional[str], aligned_group: int = 0):
        """
        Constructs an ImageStack object from a URL and a base URL.

        The following examples will all load from the same location:

        - url: https://www.example.com/images/primary_images.json  baseurl: None
        - url: https://www.example.com/images/primary_images.json  baseurl: I_am_ignored
        - url: primary_images.json  baseurl: https://www.example.com/images
        - url: images/primary_images.json  baseurl: https://www.example.com

        Parameters
        ----------
        url : str
            Either an absolute URL or a relative URL referring to the image to be read.
        baseurl : Optional[str]
            If url is a relative URL, then this must be provided.  If url is an absolute URL, then
            this parameter is ignored.
        aligned_group: int
            Which aligned tile group to load into the Imagestack, only applies if the
            tileset is unaligned. Default 0 (the first group)

        Returns
        -------
        ImageStack :
            An ImageStack representing encapsulating the data from the TileSet.
        """
        config = StarfishConfig()
        tileset = Reader.parse_doc(url, baseurl, backend_config=config.slicedimage)
        coordinate_groups = CropParameters.parse_coordinate_groups(tileset)
        crop_params = coordinate_groups[aligned_group]
        return cls.from_tileset(tileset, crop_parameters=crop_params)

    @classmethod
    def from_path_or_url(cls, url_or_path: str, aligned_group: int = 0) -> "ImageStack":
        """
        Constructs an ImageStack object from an absolute URL or a filesystem path.

        The following examples will all load from the same location:

        - url_or_path: file:///Users/starfish-user/images/primary_images.json
        - url_or_path: /Users/starfish-user/images/primary_images.json

        Parameters
        ----------
        url_or_path : str
            Either an absolute URL or a filesystem path to an imagestack.
        aligned_group: int
            Which aligned tile group to load into the Imagestack, only applies if the
            tileset is unaligned. Default 0 (the first group)
        """
        config = StarfishConfig()
        _, relativeurl, baseurl = resolve_path_or_url(url_or_path,
                                                      backend_config=config.slicedimage)
        return cls.from_url(relativeurl, baseurl, aligned_group)

    @classmethod
    def from_numpy(
            cls,
            array: np.ndarray,
            index_labels: Optional[Mapping[Axes, Sequence[int]]]=None,
            coordinates: Optional[xr.DataArray]=None,
    ) -> "ImageStack":
        """Create an ImageStack from a 5d numpy array with shape (n_round, n_ch, n_z, y, x)

        Parameters
        ----------
        array : np.ndarray
            5-d tensor of shape (n_round, n_ch, n_z, y, x)
        index_labels : Optional[Mapping[Axes, Sequence[int]]]
            Mapping from axes (r, ch, z) to their labels.  If this is not provided, then the axes
            will be labeled from 0..(n-1), where n=the size of the axes.
        coordinates : Optional[xr.DataArray]
            DataArray indexed by r, ch, z, with xmin, xmax, ymin, ymax, zmin, zmax as columns.  If
            this is not provided, then the ImageStack gets fake coordinates.

        Returns
        -------
        ImageStack :
            array data stored as an ImageStack

        """
        if len(array.shape) != 5:
            raise ValueError('a 5-d tensor with shape (n_round, n_ch, n_z, y, x) must be provided.')
        try:
            cls._validate_data_dtype_and_range(array)
        except TypeError:
            warnings.warn(f"ImageStack detected as {array.dtype}. Converting to float32...")
            array = img_as_float32(array)

        n_round, n_ch, n_z, height, width = array.shape

        if index_labels is None:
            index_labels = {
                Axes.ROUND: list(range(n_round)),
                Axes.CH: list(range(n_ch)),
                Axes.ZPLANE: list(range(n_z)),
            }
        else:
            assert len(index_labels[Axes.ROUND]) == n_round
            assert len(index_labels[Axes.CH]) == n_ch
            assert len(index_labels[Axes.ZPLANE]) == n_z

        tile_data = NumpyData(array, index_labels, coordinates)
        return cls(tile_data)

    @property
    def xarray(self) -> xr.DataArray:
        """Retrieves the image data as an :py:class:`xarray.DataArray`"""
        return self._data.data

    def sel(self, indexers: Mapping[Axes, Union[int, tuple]]):
        """Given a dictionary mapping the index name to either a value or a range represented as a
        tuple, return an Imagestack with each dimension indexed accordingly

        Parameters
        ----------
        indexers : Mapping[Axes, Union[int, tuple]]
            A dictionary of dim:index where index is the value or range to index the dimension

        Examples
        --------

        Create an Imagestack :py:func:`~starfish.imagestack.imagestack.ImageStack.synthetic_stack`
            >>> from starfish import ImageStack
            >>> from starfish.core.imagestack.test.factories import synthetic_stack
            >>> from starfish.types import Axes
            >>> stack = synthetic_stack(5, 5, 15, 200, 200)
            >>> stack
            <starfish.ImageStack (r: 5, c: 5, z: 15, y: 200, x: 200)>
            >>> stack.sel({Axes.ROUND: (1, None), Axes.CH: 0, Axes.ZPLANE: 0})
            <starfish.ImageStack (r: 4, c: 1, z: 1, y: 200, x: 200)>
            >>> stack.sel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 1,
            ...Axes.Y: 100, Axes.X: (None, 100)})
            <starfish.ImageStack (r: 1, c: 1, z: 1, y: 1, x: 100)>
            and the imagestack's physical coordinates
            xarray also indexed and recalculated according to the x,y slicing.

        Returns
        -------
        ImageStack :
            a new image stack indexed by given value or range.
        """
        stack = deepcopy(self)
        selector = indexing_utils.convert_to_selector(indexers)
        stack._data._data = indexing_utils.index_keep_dimensions(self.xarray, selector)
        return stack

    def isel(self, indexers: Mapping[Axes, Union[int, tuple]]):
        """Given a dictionary mapping the index name to either a value or a range represented as a
        tuple, return an Imagestack with each dimension indexed by position accordingly

        Parameters
        ----------
        indexers : Dict[Axes, (int/tuple)]
            A dictionary of dim:index where index is the value or range to index the dimension

        Examples
        --------

        Create an Imagestack using the ``synthetic_stack`` method
            >>> from starfish import ImageStack
            >>> from starfish.types import Axes
            >>> stack = ImageStack.synthetic_stack(5, 5, 15, 200, 200)
            >>> stack
            <starfish.ImageStack (r: 5, c: 5, z: 15, y: 200, x: 200)>
            >>> stack.isel({Axes.ROUND: (1, None), Axes.CH: 0, Axes.ZPLANE: 0})
            <starfish.ImageStack (r: 4, c: 1, z: 1, y: 200, x: 200)>
            >>> stack.isel({Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: 1,
            ...Axes.Y: 100, Axes.X: (None, 100)})
            <starfish.ImageStack (r: 1, c: 1, z: 1, y: 1, x: 100)>
            and the imagestack's physical coordinates
            xarray also indexed and recalculated according to the x,y slicing.

        Returns
        -------
        ImageStack :
            a new image stack indexed by given value or range.
        """
        stack = deepcopy(self)
        selector = indexing_utils.convert_to_selector(indexers)
        stack._data._data = indexing_utils.index_keep_dimensions(self.xarray, selector, by_pos=True)
        return stack

    def sel_by_physical_coords(
            self, indexers: Mapping[Coordinates, Union[Number, Tuple[Number, Number]]]):
        """
        Given a dictionary mapping the coordinate name to either a value or a range represented as a
        tuple, return an Imagestack with each the Coordinate dimension indexed accordingly.

        Parameters
        ----------
        indexers : Mapping[Coordinates, Union[Number, Tuple[Number, Number]]]:
            A dictionary of coord:index where index is the value or range to index the coordinate
            dimension.

        Returns
        -------
        ImageStack :
            a new image stack indexed by given value or range.
        """
        new_indexers = indexing_utils.convert_coords_to_indices(self.xarray, indexers)
        return self.isel(new_indexers)

    def get_slice(
            self,
            selector: Mapping[Axes, Union[int, slice]]
    ) -> Tuple[np.ndarray, Sequence[Axes]]:
        """
        Given a dictionary mapping the index name to either a value or a slice range, return a
        numpy array representing the slice, and a list of the remaining axes beyond the normal x-y
        tile.

        Examples
        --------

        Slicing with a scalar
            >>> from starfish import ImageStack
            >>> from starfish.core.imagestack.test.factories import synthetic_stack
            >>> from starfish.types import Axes
            >>> stack = synthetic_stack(3, 4, 5, 20, 10)
            >>> stack.shape
            OrderedDict([(<Axes.ROUND: 'r'>, 3),
             (<Axes.CH: 'c'>, 4),
             (<Axes.ZPLANE: 'z'>, 5),
             ('y', 20),
             ('x', 10)])
            >>> stack.axis_labels(Axes.ROUND)
            [0, 1, 2]
            >>> stack.axis_labels(Axes.CH)
            [0, 1, 2, 3]
            >>> stack.axis_labels(Axes.ZPLANE)
            [2, 3, 4, 5, 6]
            >>> data, axes = stack.get_slice({Axes.ZPLANE: 6})
            >>> data.shape
            (3, 4, 20, 10)
            >>> axes
            [<Axes.ROUND: 'r'>, <Axes.CH: 'c'>]

        Slicing with a range
            >>> from starfish import ImageStack
            >>> from starfish.core.imagestack.test.factories import synthetic_stack
            >>> from starfish.types import Axes
            >>> stack = synthetic_stack(3, 4, 5, 20, 10)
            >>> stack.shape
            OrderedDict([(<Axes.ROUND: 'r'>, 3),
             (<Axes.CH: 'c'>, 4),
             (<Axes.ZPLANE: 'z'>, 5),
             ('y', 20),
             ('x', 10)])
            >>> stack.axis_labels(Axes.ROUND)
            [0, 1, 2]
            >>> stack.axis_labels(Axes.CH)
            [0, 1, 2, 3]
            >>> stack.axis_labels(Axes.ZPLANE)
            [2, 3, 4, 5, 6]
            >>> data, axes = stack.get_slice({Axes.ZPLANE: 5, Axes.CH: slice(2, 4)})
            >>> data.shape
            (3, 2, 20, 10)
            >>> axes
            [<Axes.ROUND: 'r'>, <Axes.CH: 'c'>]
        """
        formatted_indexers = indexing_utils.convert_to_selector(selector)
        _, axes = self._build_slice_list(selector)
        result = self._data.sel(formatted_indexers).values

        if result.dtype != np.float32:
            warnings.warn(
                f"Non-float32 dtype: {result.dtype} detected. Data has likely been set using "
                f"private attributes of ImageStack. ImageStack only supports float data in the "
                f"range [0, 1]. Many algorithms will not function properly if provided other "
                f"DataTypes. See: http://scikit-image.org/docs/dev/user_guide/data_types.html")

        return result, axes

    def set_slice(
            self,
            selector: Mapping[Axes, Union[int, slice]],
            data: np.ndarray,
            axes: Optional[Sequence[Axes]]=None):
        """
        Given a dictionary mapping the index name to either a value or a slice range and a source
        numpy array, set the slice of the array of this ImageStack to the values in the source
        numpy array.

        Consumers of this API should not be aware of the internal order of the axes in ImageStack.
        As a result, they should be explicitly providing the order of the axes of the numpy array.
        This method will reorder the data to the internal order of the axes in ImageStack before
        writing it.

        Parameters
        ----------
        selector : Mapping[Axes, Union[int, slice]]
            The slice of the data we are writing with this operation.  Each index should map to a
            value or a range.  If the index is not present, we are writing to the entire range along
            that index.
        data : np.ndarray
            a 2- to 5-D numpy array containing the source data for the operation whose last two axes
            must be in (Y, X) order. If data larger than 2-D is provided, axes must be set to
            specify the order of the additional axes (see below).
        axes : Optional[Sequence[Axes]]
            The order of the axes for the source data, excluding (Y, X). Optional ONLY if data is
            a (Y, X) 2-d tile.

        Examples
        --------
        Setting a slice indicated by scalars.

            >>> import numpy as np
            >>> from starfish import ImageStack
            >>> from starfish.core.imagestack.test.factories import synthetic_stack
            >>> from starfish.types import Axes
            >>> stack = synthetic_stack(3, 4, 5, 20, 10)
            >>> stack.shape
            OrderedDict([(<Axes.ROUND: 'r'>, 3),
             (<Axes.CH: 'c'>, 4),
             (<Axes.ZPLANE: 'z'>, 5),
             ('y', 20),
             ('x', 10)])
            >>> new_data = np.zeros((3, 4, 10, 20), dtype=np.float32)
            >>> stack.set_slice(new_data, axes=[Axes.ROUND, Axes.CH]

        Setting a slice indicated by scalars.  The data presented has a different axis order than
        the previous example.

            >>> import numpy as np
            >>> from starfish import ImageStack
            >>> from starfish.core.imagestack.test.factories import synthetic_stack
            >>> from starfish.types import Axes
            >>> stack = synthetic_stack(3, 4, 5, 20, 10)
            >>> stack.shape
            OrderedDict([(<Axes.ROUND: 'r'>, 3),
             (<Axes.CH: 'c'>, 4),
             (<Axes.ZPLANE: 'z'>, 5),
             ('y', 20),
             ('x', 10)])
            >>> new_data = np.zeros((4, 3, 10, 20), dtype=np.float32)
            >>> stack.set_slice(new_data, axes=[Axes.CH, Axes.ROUND]

        Setting a slice indicated by a range.

            >>> from starfish import ImageStack
            >>> from starfish.core.imagestack.test.factories import synthetic_stack
            >>> from starfish.types import Axes
            >>> stack = synthetic_stack(3, 4, 5, 20, 10)
            >>> stack.shape
            OrderedDict([(<Axes.ROUND: 'r'>, 3),
             (<Axes.CH: 'c'>, 4),
             (<Axes.ZPLANE: 'z'>, 5),
             ('y', 20),
             ('x', 10)])
            >>> new_data = np.zeros((3, 2, 10, 20), dtype=np.float32)
            >>> stack.set_slice({Axes.ZPLANE: 5, Axes.CH: slice(2, 4)}, new_data)
        """

        self._validate_data_dtype_and_range(data)

        slice_list, expected_axes = self._build_slice_list(selector)

        if axes is None:
            axes = list()
        if len(axes) != len(data.shape) - 2:
            raise ValueError(
                "data shape ({}) should be the axes ({}) and (Y,X).".format(data.shape, axes))
        move_src = list()
        move_dst = list()
        for src_idx, axis in enumerate(axes):
            try:
                dst_idx = expected_axes.index(axis)
            except ValueError:
                raise ValueError(
                    "Unexpected axis {}.  Expecting only {}.".format(axis, expected_axes))
            if src_idx != dst_idx:
                move_src.append(src_idx)
                move_dst.append(dst_idx)

        if len(move_src) != 0:
            data = np.moveaxis(data, move_src, move_dst)

        if self._data.loc[slice_list].shape != data.shape:
            raise ValueError("source shape {} mismatches destination shape {}".format(
                data.shape, self._data[slice_list].shape))

        self._data.loc[slice_list] = data

    @staticmethod
    def _build_slice_list(
            selector: Mapping[Axes, Union[int, slice]]
    ) -> Tuple[Tuple[Union[int, slice], ...], Sequence[Axes]]:
        slice_list: MutableSequence[Union[int, slice]] = [
            slice(None, None, None)
            for _ in range(N_AXES)
        ]
        axes = []
        removed_axes = set()
        for name, value in selector.items():
            idx = AXES_DATA[name].order
            if not isinstance(value, slice):
                removed_axes.add(name)
            slice_list[idx] = value

        for dimension_value, dimension_name in sorted([
            (dimension_value.order, dimension_name)
            for dimension_name, dimension_value in AXES_DATA.items()
        ]):
            if dimension_name not in removed_axes:
                axes.append(dimension_name)

        return tuple(slice_list), axes

    def _iter_axes(self, axes: Set[Axes]=None) -> Iterator[Mapping[Axes, int]]:
        """Iterate over provided axes.

        Parameters
        ----------
        axes : Set[Axes]
            The set of Axes to be iterated over (default={Axes.ROUND, Axes.CH}).

        Yields
        ------
        Dict[str, int]
            Mapping of axis name to index

        """
        if axes is None:
            axes = {Axes.ROUND, Axes.CH}
        ordered_axes = list(axes)
        ranges = [self.axis_labels(ind) for ind in ordered_axes]
        for items in product(*ranges):
            a = zip(ordered_axes, items)
            yield {ind: val for (ind, val) in a}

    def apply(
            self,
            func: Callable,
            group_by: Set[Axes]=None,
            in_place=False,
            verbose: bool=False,
            n_processes: Optional[int]=None,
            clip_method: Union[str, Clip]=Clip.CLIP,
            **kwargs
    ) -> "ImageStack":
        """Split the image along a set of axes and apply a function across all the components.  This
        function should yield data of the same dimensionality as the input components.  These
        resulting components are then constituted into an ImageStack and returned.

        Parameters
        ----------
        func : Callable
            Function to apply. must expect a first argument which is a 2d or 3d numpy array and
            return an array of the same shape.
        group_by : Set[Axes]
            Axes to split the data along.
            `ex. splitting a 2D array (axes: X, Y; size:
            3, 4) by X results in 3 arrays of size 4.  (default {Axes.ROUND, Axes.CH,
            Axes.ZPLANE})`
        in_place : bool
            If True, function is executed in place. If n_proc is not 1, the tile or
            volume will be copied once during execution. If false, a new ImageStack object will be
            produced. (Default False)
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        n_processes : Optional[int]
            The number of processes to use for apply. If None, uses the output of os.cpu_count()
            (default = None).
        kwargs : dict
            Additional arguments to pass to func
        clip_method : Union[str, :py:class:`~starfish.types.Clip`]

            - Clip.CLIP (default): Controls the way that data are scaled to retain skimage dtype
              requirements that float data fall in [0, 1].
            - Clip.CLIP: data above 1 are set to 1, and below 0 are set to 0
            - Clip.SCALE_BY_IMAGE: data above 1 are scaled by the maximum value, with the maximum
              value calculated over the entire ImageStack
            - Clip.SCALE_BY_CHUNK: data above 1 are scaled by the maximum value, with the maximum
              value calculated over each slice, where slice shapes are determined by the group_by
              parameters


        Returns
        -------
        ImageStack :
            If inplace is False, return a new ImageStack, otherwise return a reference to the
            original stack with data modified by application of func

        Raises
        ------
        TypeError :
             If no Clip method given.

        """
        # default grouping is by (x, y) tile
        if group_by is None:
            group_by = {Axes.ROUND, Axes.CH, Axes.ZPLANE}

        if not isinstance(clip_method, (str, Clip)):
            raise TypeError("must pass a Clip method. See starfish.types.Clip for valid options")

        if not in_place:
            # create a copy of the ImageStack, call apply on that stack with in_place=True
            image_stack = deepcopy(self)
            return image_stack.apply(
                func=func,
                group_by=group_by, in_place=True, verbose=verbose, n_processes=n_processes,
                clip_method=clip_method,
                **kwargs
            )

        # wrapper adds a target `data` parameter where the results from func will be stored
        # data are clipped or scaled by chunk using preserve_float_range if clip_method != 2
        bound_func = partial(ImageStack._in_place_apply, func, clip_method=clip_method)

        # execute the processing workflow
        self.transform(
            func=bound_func,
            group_by=group_by,
            verbose=verbose,
            n_processes=n_processes,
            **kwargs)

        # scale based on values of whole image
        if clip_method == Clip.SCALE_BY_IMAGE:
            self._data.data.values = preserve_float_range(self._data.data.values, rescale=True)

        return self

    @staticmethod
    def _in_place_apply(
        apply_func: Callable[..., Union[xr.DataArray, np.ndarray]], data: np.ndarray,
        clip_method: Union[str, Clip], **kwargs
    ) -> None:
        result = apply_func(data, **kwargs)
        if clip_method == Clip.CLIP:
            data[:] = preserve_float_range(result, rescale=False)
        elif clip_method == Clip.SCALE_BY_CHUNK:
            data[:] = preserve_float_range(result, rescale=True)
        else:
            data[:] = result

    def transform(
            self,
            func: Callable,
            group_by: Set[Axes]=None,
            verbose=False,
            n_processes: Optional[int]=None,
            **kwargs
    ) -> List[Any]:
        """Split the image along a set of axes, and apply a function across all the components.

        Parameters
        ----------
        func : Callable
            Function to apply. must expect a first argument which is a numpy array (see group_by)
            but may return any object type.
        group_by : Set[Axes]
            Axes to split the data along.  For instance, splitting a 2D array (axes: X, Y; size:
            3, 4) by X results in 3 arrays of size 4.  (default {Axes.ROUND, Axes.CH,
            Axes.ZPLANE})
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        n_processes : Optional[int]
            The number of processes to use for apply. If None, uses the output of os.cpu_count()
            (default = None).
        kwargs : dict
            Additional arguments to pass to func being applied

        Returns
        -------
        List[Any] :
            The results of applying func to stored image data
        """
        # default grouping is by (x, y) tile
        if group_by is None:
            group_by = {Axes.ROUND, Axes.CH, Axes.ZPLANE}

        selectors = list(self._iter_axes(group_by))
        slice_lists = [self._build_slice_list(index)[0]
                       for index in selectors]

        selectors_and_slice_lists = zip(selectors, slice_lists)
        if verbose and StarfishConfig().verbose:
            selectors_and_slice_lists = tqdm(selectors_and_slice_lists)

        coordinates = {
            dim: self.xarray.coords[dim]
            for dim in self.xarray.coords.dims
        }

        mp_applyfunc: Callable = partial(
            self._processing_workflow, partial(func, **kwargs), self.xarray.dims, coordinates)

        with Pool(
                processes=n_processes,
                initializer=SharedMemory.initializer,
                initargs=((self._data._backing_mp_array,
                           self._data._data.shape,
                           self._data._data.dtype),)) as pool:
            results = pool.imap(mp_applyfunc, selectors_and_slice_lists)

            # Note: results is [None, ...] if executing an in-place workflow
            # Note: this return must be inside the context manager or the Pool will deadlock
            return list(zip(results, selectors))

    @staticmethod
    def _processing_workflow(
            worker_callable: Callable[[np.ndarray], Any],
            xarray_dims: Sequence[str],
            xarray_coordinates: Mapping[str, np.ndarray],
            selector_and_slice_list: Tuple[Mapping[Axes, int],
                                           Tuple[Union[int, slice], ...]],
    ):
        # build the numpy array from the shared memory object
        backing_mp_array, shape, dtype = SharedMemory.get_payload()
        unshaped_numpy_array = np.frombuffer(backing_mp_array.get_obj(), dtype=dtype)

        # build and then slice the xarray to get the piece needed for this worker
        data_array = xr.DataArray(
            data=unshaped_numpy_array.reshape(shape),
            dims=xarray_dims,
            coords=xarray_coordinates,
        )
        sliced = data_array.sel(selector_and_slice_list[0])

        # pass worker_callable a view into the backing array, which will be overwritten
        return worker_callable(sliced)  # type: ignore

    @property
    def tile_metadata(self) -> pd.DataFrame:
        """return a table containing Tile metadata

        Returns
        -------
        pd.DataFrame :
            dataframe containing per-tile metadata information for each image. Guaranteed to
            include information on channel, imaging round, z plane, and barcode index. Also
            contains any information stored in the extras field for each tile.

        """

        data: collections.defaultdict = collections.defaultdict(list)
        keys = self._tile_data.keys()
        index_keys = set(
            key.value
            for key in AXES_DATA.keys()
        )
        extras_keys = set(
            key
            for tilekey in keys
            for key in self._tile_data[tilekey].keys())
        duplicate_keys = index_keys.intersection(extras_keys)
        if len(duplicate_keys) > 0:
            duplicate_keys_str = ", ".join([str(key) for key in duplicate_keys])
            raise ValueError(
                f"keys ({duplicate_keys_str}) was found in both the Tile specification and extras "
                f"field. Tile specification keys may not be duplicated in the extras field.")

        for selector in self._iter_axes({Axes.ROUND, Axes.CH, Axes.ZPLANE}):
            tilekey = TileKey(
                round=selector[Axes.ROUND],
                ch=selector[Axes.CH],
                zplane=selector[Axes.ZPLANE])
            extras = self._tile_data[tilekey]

            for index, index_value in selector.items():
                data[index.value].append(index_value)

            for k in extras_keys:
                data[k].append(extras.get(k, None))

            if 'barcode_index' not in extras:
                barcode_index = ((((selector[Axes.ZPLANE]
                                    * self.num_rounds) + selector[Axes.ROUND])
                                  * self.num_chs) + selector[Axes.CH])

                data['barcode_index'].append(barcode_index)

        return pd.DataFrame(data)

    @property
    def log(self) -> List[dict]:
        """
        Returns a list of pipeline components that have been applied to this imagestack
        as well as their corresponding runtime parameters.

        For more information about provenance logging see
        `Provenance Logging
        <https://spacetx-starfish.readthedocs.io/en/latest/help_and_reference/api/utils/ilogging.html>`_

        Returns
        -------
        List[dict]
        """
        return self._log

    def update_log(self, class_instance) -> None:
        """
        Adds a new entry to the log list.

        Parameters
        ----------
        class_instance: The instance of a class being applied to the imagestack
        """
        entry = {"method": class_instance.__class__.__name__,
                 "arguments": class_instance.__dict__,
                 "os": logging.get_os_info(),
                 "dependencies": logging.get_core_dependency_info(),
                 "release tag": logging.get_release_tag(),
                 "starfish version": logging.get_dependency_version('starfish')
                 }
        self._log.append(entry)

    @property
    def raw_shape(self) -> Tuple[int, int, int, int, int]:
        """
        Returns the shape of the 5-d image tensor stored as self.image

        Returns
        -------
        Tuple[int, int, int, int, int] :
            The size of the image tensor
        """
        return self._data.shape

    @property
    def shape(self) -> collections.OrderedDict:
        """
        Returns the shape of the space that this image inhabits.  It does not include the
        dimensions of the image itself.  For instance, if this is an X-Y image in a C-H-Y-X space,
        then the shape would include the axes C and H.

        Returns
        -------
        An ordered mapping between index names to the size of the index.
        """
        # TODO: (ttung) Note that the return type should be ..OrderedDict[Any, str], but python3.6
        # has a bug where this # breaks horribly.  Can't find a bug id to link to, but see
        # https://stackoverflow.com/questions/41207128/how-do-i-specify-ordereddict-k-v-types-for-\
        # mypy-type-annotation
        result: collections.OrderedDict[Any, str] = collections.OrderedDict()
        for name, data in AXES_DATA.items():
            result[name] = self._data.shape[data.order]
        result['y'] = self._data.shape[-2]
        result['x'] = self._data.shape[-1]

        return result

    @property
    def num_rounds(self):
        """Return the number of rounds in the ImageStack"""
        return self.xarray.sizes[Axes.ROUND]

    @property
    def num_chs(self):
        """Return the number of channels in the ImageStack"""
        return self.xarray.sizes[Axes.CH]

    @property
    def num_zplanes(self):
        """Return the number of z_planes in the ImageStack"""
        return self.xarray.sizes[Axes.ZPLANE]

    def axis_labels(self, axis: Axes) -> Sequence[int]:
        """Given an axis, return the sorted unique values for that axis in this ImageStack.  For
        instance, ``imagestack.axis_labels(Axes.ROUND)`` returns all the round ids in this
        imagestack."""

        return [int(val) for val in self.xarray.coords[axis.value].values]

    @property
    def tile_shape(self):
        """Return the shape of each tile in the ImageStack. All
        Tiles have the same shape."""
        return self.xarray.sizes[Axes.Y], self.xarray.sizes[Axes.X]

    def to_multipage_tiff(self, filepath: str) -> None:
        """save the ImageStack as a FIJI-compatible multi-page TIFF file

        Parameters
        ----------
        filepath : str
            filepath for a tiff FILE. "TIFF" suffix will be added if the provided path does not
            end with .TIFF

        """
        if not filepath.upper().endswith(".TIFF"):
            filepath += ".TIFF"

        # RZCYX is the order expected by FIJI
        data = self.xarray.transpose(
            Axes.ROUND.value,
            Axes.ZPLANE.value,
            Axes.CH.value,
            Axes.Y.value,
            Axes.X.value)

        # Any float32 image with low dynamic range will provoke a warning that the image is
        # low contrast because the data must be converted to uint16 for compatibility with FIJI.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            skimage.io.imsave(filepath, data.values, imagej=True)

    def export(self,
               filepath: str,
               tile_opener=None,
               tile_format: ImageFormat=ImageFormat.NUMPY) -> None:
        """write the image tensor to disk in spaceTx format

        Parameters
        ----------
        filepath : str
            Path + prefix for the images and primary_images.json written by this function
        tile_opener : TODO ttung: doc me.
        tile_format : ImageFormat
            Format in which each 2D plane should be written.

        """
        # Add log data to extras
        self._tile_data.extras[STARFISH_EXTRAS_KEY] = logging.LogEncoder().encode({LOG: self.log})
        tileset = TileSet(
            dimensions={
                Axes.ROUND,
                Axes.CH,
                Axes.ZPLANE,
                Axes.Y,
                Axes.X,
            },
            shape={
                Axes.ROUND: self.num_rounds,
                Axes.CH: self.num_chs,
                Axes.ZPLANE: self.num_zplanes,
            },
            default_tile_shape={Axes.Y: self.tile_shape[0], Axes.X: self.tile_shape[1]},
            extras=self._tile_data.extras,
        )
        for axis_val_map in self._iter_axes({Axes.ROUND, Axes.CH, Axes.ZPLANE}):
            tilekey = TileKey(
                round=axis_val_map[Axes.ROUND],
                ch=axis_val_map[Axes.CH],
                zplane=axis_val_map[Axes.ZPLANE])
            round_, ch, zplane = tilekey.round, tilekey.ch, tilekey.z
            extras: dict = self._tile_data[tilekey]

            selector = {
                Axes.ROUND: round_,
                Axes.CH: ch,
                Axes.ZPLANE: zplane,
            }

            coordinates: MutableMapping[Coordinates, Union[Tuple[Number, Number], Number]] = dict()
            x_coordinates = (float(self.xarray[Coordinates.X.value][0]),
                             float(self.xarray[Coordinates.X.value][-1]))
            y_coordinates = (float(self.xarray[Coordinates.Y.value][0]),
                             float(self.xarray[Coordinates.Y.value][-1]))
            coordinates[Coordinates.X] = x_coordinates
            coordinates[Coordinates.Y] = y_coordinates
            if Coordinates.Z in self.xarray.coords:
                # set the z coord to the calculated value from the associated z plane
                z_coordinates = float(self.xarray[Coordinates.Z.value][zplane])
                coordinates[Coordinates.Z] = z_coordinates

            tile = Tile(
                coordinates=coordinates,
                indices=selector,
                extras=extras,
            )
            tile.numpy_array, _ = self.get_slice(
                selector={Axes.ROUND: round_, Axes.CH: ch, Axes.ZPLANE: zplane}
            )
            tileset.add_tile(tile)

        if tile_opener is None:
            def tile_opener(tileset_path: Path, tile, ext):
                base = tileset_path.parent / tileset_path.stem
                if Axes.ZPLANE in tile.indices:
                    zval = tile.indices[Axes.ZPLANE]
                    zstr = "-Z{}".format(zval)
                else:
                    zstr = ""
                return open(
                    "{}-H{}-C{}{}.{}".format(
                        str(base),
                        tile.indices[Axes.ROUND],
                        tile.indices[Axes.CH],
                        zstr,
                        ext,
                    ),
                    "wb")

        if not filepath.endswith('.json'):
            filepath += '.json'
        Writer.write_to_path(
            tileset,
            filepath,
            pretty=True,
            tile_opener=tile_opener,
            tile_format=tile_format)

    def max_proj(self, *dims: Axes) -> "ImageStack":
        """return a max projection over one or more axis of the image tensor

        Parameters
        ----------
        dims : Axes
            one or more axes to project over

        Returns
        -------
        np.ndarray :
            max projection

        """
        max_projection = self._data.max([dim.value for dim in dims])
        max_projection = max_projection.expand_dims(tuple(dim.value for dim in dims))
        max_projection = max_projection.transpose(*self.xarray.dims)
        max_proj_stack = self.from_numpy(max_projection.values)
        return max_proj_stack

    def _squeezed_numpy(self, *dims: Axes):
        """return this ImageStack's data as a squeezed numpy array"""
        return self.xarray.squeeze(tuple(dim.value for dim in dims)).values
