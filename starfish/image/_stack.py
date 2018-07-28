import collections
import os
import warnings
from copy import deepcopy
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, Iterator, List, Mapping, MutableSequence, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import scoreatpercentile
from skimage import exposure
from slicedimage import Reader, Writer, TileSet, Tile
from slicedimage.io import resolve_path_or_url
from tqdm import tqdm

from starfish.constants import Coordinates, Indices, Features
from starfish.errors import DataFormatWarning
from starfish.intensity_table import IntensityTable
from starfish.pipeline.features.spot_attributes import SpotAttributes


_DimensionMetadata = collections.namedtuple("_DimensionMetadata", ['order', 'required'])


class ImageStack:
    """Container for a TileSet (field of view)

    Methods
    -------
    get_slice    retrieve a slice of the image tensor
    set_slice    set a slice of the image tensor
    apply        apply a 2d or 3d function across all Tiles in the image tensor
    max_proj     return a max projection over one or more axis of the image tensor
    show_stack   show an interactive, pageable view of the image tensor, or a slice of the image tensor
    write        save the (potentially modified) image tensor to disk

    Properties
    ----------
    num_chs      the number of channels stored in the image tensor
    num_rounds   the number of imaging rounds stored in the image tensor
    num_zlayers  the number of z-layers stored in the image tensor
    numpy_array  the 5-d image tensor is stored in this array
    raw_shape    the shape of the image tensor (in integers)
    shape        the shape of the image tensor by categorical index (channels, imaging rounds, z-layers)
    """

    AXES_DATA: Mapping[Indices, _DimensionMetadata] = {
        Indices.ROUND: _DimensionMetadata(0, True),
        Indices.CH: _DimensionMetadata(1, True),
        Indices.Z: _DimensionMetadata(2, False),
    }
    N_AXES = max(data.order for data in AXES_DATA.values()) + 1

    def __init__(self, image_partition):
        self._image_partition = image_partition
        self._tile_shape = image_partition.default_tile_shape

        # Examine the tiles to figure out the right kind (int, float, etc.) and size.  We require that all the tiles
        # have the same kind of data type, but we do not require that they all have the same size of data type.  The
        # allocated array is the highest size we encounter.
        kind = None
        max_size = 0
        for tile in self._image_partition.tiles():
            dtype = tile.numpy_array.dtype
            if kind is None:
                kind = dtype.kind
            else:
                if kind != dtype.kind:
                    raise TypeError("All tiles should have the same kind of dtype")
            if dtype.itemsize > max_size:
                max_size = dtype.itemsize
            if self._tile_shape is None:
                self._tile_shape = tile.tile_shape
            elif tile.tile_shape is not None and self._tile_shape != tile.tile_shape:
                raise ValueError("Starfish does not support tiles that are not identical in shape")

        shape = [
            self._get_dimension_size(axis_name)
            for axis_name, axis_data in ImageStack.AXES_DATA.items()
            for ix in range(ImageStack.N_AXES)
            if ix == axis_data.order
        ]
        shape.extend(self._tile_shape)

        # now that we know the tile data type (kind and size), we can allocate the data array.
        self._data = np.zeros(shape=shape, dtype=np.dtype(f"{kind}{max_size}"))

        # iterate through the tiles and set the data.
        for tile in self._image_partition.tiles():
            h = tile.indices[Indices.ROUND]
            c = tile.indices[Indices.CH]
            zlayer = tile.indices.get(Indices.Z, 0)
            data = tile.numpy_array
            if max_size != data.dtype.itemsize:
                if data.dtype.kind == "i" or data.dtype.kind == "u":
                    # fixed point can be done with a simple multiply.
                    src_range = np.iinfo(data.dtype).max - np.iinfo(data.dtype).min + 1
                    dst_range = np.iinfo(self._data.dtype).max - np.iinfo(self._data.dtype).min + 1
                    data = data * (dst_range / src_range)
                warnings.warn(
                    f"Tile "
                    f"(R: {tile.indices[Indices.ROUND]} C: {tile.indices[Indices.CH]} "
                    f"Z: {tile.indices[Indices.Z]}) has "
                    f"dtype {data.dtype}.  One or more tiles is of a larger dtype {self._data.dtype}.",
                    DataFormatWarning)
            self.set_slice(indices={Indices.ROUND: h, Indices.CH: c, Indices.Z: zlayer}, data=data)
        # set_slice will mark the data as needing writeback, so we need to unset that.
        self._data_needs_writeback = False

    @classmethod
    def from_url(cls, url: str, baseurl: Optional[str]):
        """
        Constructs an ImageStack object from a URL and a base URL.

        The following examples will all load from the same location:
          1. url: https://www.example.com/images/hybridization.json  baseurl: None
          2. url: https://www.example.com/images/hybridization.json  baseurl: I_am_ignored
          3. url: hybridization.json  baseurl: https://www.example.com/images
          4. url: images/hybridization.json  baseurl: https://www.example.com

        Parameters:
        -----------
        url : str
            Either an absolute URL or a relative URL referring to the image to be read.
        baseurl : Optional[str]
            If url is a relative URL, then this must be provided.  If url is an absolute URL, then this parameter is
            ignored.
        """
        image_partition = Reader.parse_doc(url, baseurl)

        return cls(image_partition)

    @classmethod
    def from_path_or_url(cls, url_or_path: str) -> "ImageStack":
        """
        Constructs an ImageStack object from an absolute URL or a filesystem path.

        The following examples will all load from the same location:
          1. url_or_path: file:///Users/starfish-user/images/hybridization.json
          2. url_or_path: /Users/starfish-user/images/hybridization.json

        Parameters:
        -----------
        url_or_path : str
            Either an absolute URL or a filesystem path to an imagestack.
        """
        _, relativeurl, baseurl = resolve_path_or_url(url_or_path)
        return cls.from_url(relativeurl, baseurl)

    @property
    def numpy_array(self):
        """Retrieves a view of the image data as a numpy array."""
        result = self._data.view()
        result.setflags(write=False)
        return result

    @numpy_array.setter
    def numpy_array(self, data):
        """Sets the image's data from a numpy array.  The numpy array is advised to be immutable afterwards."""
        self._data = data.view()
        self._data_needs_writeback = True
        data.setflags(write=False)

    def get_slice(
            self,
            indices: Mapping[Indices, Union[int, slice]]
    ) -> Tuple[np.ndarray, Sequence[Indices]]:
        """
        Given a dictionary mapping the index name to either a value or a slice range, return a numpy array representing
        the slice, and a list of the remaining axes beyond the normal x-y tile.

        Example:
            ImageStack axes: H, C, and Z with shape 3, 4, 5, respectively.
            ImageStack Implicit axes: X, Y with shape 10, 20, respectively.
            Called to slice with indices {Z: 5}.
            Result: a 4-dimensional numpy array with shape (3, 4, 20, 10) and the remaining axes [H, C].

        Example:
            Original axes: H, C, and Z.
            Implicit axes: X, Y.
            Called to slice with indices {Z: 5, C: slice(2, 4)}.
            Result: a 4-dimensional numpy array with shape (3, 2, 20, 10) and the remaining axes [H, C].
        """
        slice_list, axes = self._build_slice_list(indices)
        result = self._data[slice_list]
        result.setflags(write=False)
        return result, axes

    def set_slice(
            self,
            indices: Mapping[Indices, Union[int, slice]],
            data: np.ndarray,
            axes: Sequence[Indices]=None):
        """
        Given a dictionary mapping the index name to either a value or a slice range and a source numpy array, set the
        slice of the array of this ImageStack to the values in the source numpy array.  If the optional parameter axes
        is provided, that represents the axes of the numpy array beyond the x-y tile.

        Example:
            ImageStack axes: H, C, and Z with shape 3, 4, 5, respectively.
            ImageStack Implicit axes: X, Y with shape 10, 20, respectively.
            Called to set a slice with indices {Z: 5}.
            Data: a 4-dimensional numpy array with shape (3, 4, 10, 20)
            Result: Replace the data for Z=5.

        Example:
            ImageStack axes: H, C, and Z. (shape 3, 4, 5)
            ImageStack Implicit axes: X, Y. (shape 10, 20)
            Called to set a slice with indices {Z: 5, C: slice(2, 4)}.
            Data: a 4-dimensional numpy array with shape (3, 2, 10, 20)
            Result: Replace the data for Z=5, C=2-3.
        """
        slice_list, expected_axes = self._build_slice_list(indices)

        if axes is not None:
            if len(axes) != len(data.shape) - 2:
                raise ValueError("data shape ({}) should be the axes ({}) and (x,y).".format(data.shape, axes))
            move_src = list()
            move_dst = list()
            for src_idx, axis in enumerate(axes):
                try:
                    dst_idx = expected_axes.index(axis)
                except ValueError:
                    raise ValueError("Unexpected axis {}.  Expecting only {}.".format(axis, expected_axes))
                if src_idx != dst_idx:
                    move_src.append(src_idx)
                    move_dst.append(dst_idx)

            if len(move_src) != 0:
                data = data.view()
                np.moveaxis(data, move_src, move_dst)

        if self._data[slice_list].shape != data.shape:
            raise ValueError("source shape {} mismatches destination shape {}".format(
                data.shape, self._data[slice_list].shape))

        self._data[slice_list] = data
        self._data_needs_writeback = True

    # TODO ambrosejcarr: update to use IntensityTable instead of SpotAttributes
    def show_stack(
            self, indices: Mapping[Indices, Union[int, slice]],
            color_map: str= 'gray', figure_size: Tuple[int, int]=(10, 10),
            show_spots: Optional[SpotAttributes]=None,
            rescale: bool=False, p_min: Optional[float]=None, p_max: Optional[float]=None, **kwargs):
        """Create an interactive visualization of an image stack

        Produces a slider that flips through the selected volume tile-by-tile. Supports manual adjustment of dynamic
        range.

        Parameters
        ----------
        indices : Mapping[Indices, Union[int, slice]],
            Indices to select a volume to visualize. Passed to `Image.get_slice()`.
            See `Image.get_slice()` for examples.
        color_map : str (default = 'gray')
            string id of a matplotlib colormap
        figure_size : Tuple[int, int] (default = (10, 10))
            size of the figure in inches
        show_spots : Optional[SpotAttributes]
            [Preliminary functionality] if provided, should be a SpotAttribute table that corresponds
            to the volume being displayed. This will be paired automatically in the future.
        rescale : bool (default = False)
            if True, rescale the data to exclude high and low-value outliers (see skimage.exposure.rescale_intensity).
        p_min: float
            clip values below this intensity percentile. If provided, overrides rescale, above. (default = None)
        p_max: float
            clip values above this intensity percentile. If provided, overrides rescale, above. (default = None)

        Raises
        ------
        ValueError :
            User must select one of rescale or p_min/p_max to adjust the image dynamic range. If both are selected, a
            ValueError is raised.

        Notes
        -----
        For this widget to function interactively in the notebook, after ipywidgets has been installed, the user must
        register the widget with jupyter by typing the following command into the terminal:
        jupyter nbextension enable --py widgetsnbextension

        """

        from ipywidgets import interact
        import matplotlib.pyplot as plt

        if not indices:
            raise ValueError('indices may not be an empty dict or None')

        # get the requested chunk, linearize the remaining data into a sequence of tiles
        data, remaining_inds = self.get_slice(indices)

        # identify the dimensionality of data with all dimensions other than x, y linearized
        if len(data.shape) >= 3:
            n = np.product(data.shape[:-2])
        else:
            raise ValueError(
                f'a stack with dimensionality >= 3 is required, the provided indexer produced a '
                f'stack with shape {data.shape}')

        # linearize the array
        linear_view: np.ndarray = data.reshape((n,) + data.shape[-2:])

        # set the labels for the linearized tiles
        labels: List[List[str]] = []
        for index, size in zip(remaining_inds, data.shape[:-2]):
            labels.append([f'{index}{n}' for n in range(size)])

        # mypy thinks this has an incompatible type "Iterator[Tuple[Any, ...]]";
        # it expects "Iterable[List[str]]"
        labels = list(product(*labels))  # type: ignore

        n = linear_view.shape[0]

        if rescale and any((p_min, p_max)):
            raise ValueError('select one of rescale and p_min/p_max to rescale image, not both.')

        elif rescale:
            print("Rescaling ...")
            vmin, vmax = scoreatpercentile(data, (0.5, 99.5))
            linear_view = exposure.rescale_intensity(
                linear_view,
                in_range=(vmin, vmax),
                out_range=np.float32
            ).astype(np.float32)

        elif p_min or p_max:
            print("Clipping ...")
            a_min, a_max = scoreatpercentile(
                linear_view,
                (p_min if p_min else 0, p_max if p_max else 100)
            )
            linear_view = np.clip(linear_view, a_min=a_min, a_max=a_max)

        show_spot_function = self._show_spots

        def show_plane(ax, plane, plane_index, cmap="gray", title=None):
            ax.imshow(plane, cmap=cmap)
            if show_spots:
                # this is slow. This link might have something to help:
                # https://bastibe.de/2013-05-30-speeding-up-matplotlib.html
                show_spot_function(show_spots.data, ax=ax, z=plane_index, **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])

            if title:
                ax.set_title(title)

        @interact(plane_index=(0, n - 1))
        def display_slice(plane_index=0):
            fig, ax = plt.subplots(figsize=figure_size)
            show_plane(ax, linear_view[plane_index], plane_index, title=f'{labels[plane_index]}', cmap=color_map)
            plt.show()

        return display_slice

    @staticmethod
    def _show_spots(result_df, ax, z=None, size=1, z_dist=1.5, scale_radius=5) -> None:
        """function to plot spot finding results on top of any image as hollow red circles

        called spots are colored by category

        Parameters:
        -----------
        img : np.ndarray[Any]
            2-d image of any dtype
        result_df : pd.Dataframe
            result dataframe containing spot calls that correspond to the image channel
        z : Optional[int]
            If provided, z-plane to plot spot calls for. Default (None): plot all provided spots.
        size : int
            width of line to plot around the identified spot
        z_dist : float
            plot spots if within this distance of the z-plane. Ignored if z is not passed.
        vmin, vmax : int
            clipping thresholds for the image plot
        ax, matplotlib.Axes.Axis
            axis to plot spots on

        """
        import matplotlib.pyplot as plt

        if z is not None and 'z' in result_df.columns:
            inds = np.abs(result_df['z'] - z) < z_dist
        else:
            inds = np.ones(result_df.shape[0]).astype(bool)

        # get the data needed to plot
        selected = result_df.loc[inds, ['r', 'x', 'y']]

        for i in np.arange(selected.shape[0]):
            r, x, y = selected.iloc[i, :]  # radius is a duplicate, and is present twice
            c = plt.Circle((y, x), r * scale_radius, color='r', linewidth=size, fill=False)
            ax.add_patch(c)

    @staticmethod
    def _build_slice_list(
            indices: Mapping[Indices, Union[int, slice]]
    ) -> Tuple[Tuple[Union[int, slice], ...], Sequence[Indices]]:
        slice_list: MutableSequence[Union[int, slice]] = [
            slice(None, None, None)
            for _ in range(ImageStack.N_AXES)
        ]
        axes = []
        removed_axes = set()
        for name, value in indices.items():
            idx = ImageStack.AXES_DATA[name].order
            if not isinstance(value, slice):
                removed_axes.add(name)
            slice_list[idx] = value

        for dimension_value, dimension_name in sorted([
            (dimension_value.order, dimension_name)
            for dimension_name, dimension_value in ImageStack.AXES_DATA.items()
        ]):
            if dimension_name not in removed_axes:
                axes.append(dimension_name)

        return tuple(slice_list), axes

    def _iter_indices(self, is_volume: bool=False) -> Iterator[Mapping[Indices, int]]:
        """Iterate over indices of image tiles or image volumes if is_volume is True

        Parameters
        ----------
        is_volume : bool
            If True, yield indices necessary to extract volumes from self, else return
            indices for tiles

        Yields
        ------
        Dict[str, int]
            Mapping of dimension name to index

        """
        for round_ in np.arange(self.shape[Indices.ROUND]):
            for ch in np.arange(self.shape[Indices.CH]):
                if is_volume:
                    yield {Indices.ROUND: round_, Indices.CH: ch}
                else:
                    for z in np.arange(self.shape[Indices.Z]):
                        yield {Indices.ROUND: round_, Indices.CH: ch, Indices.Z: z}

    def _iter_tiles(
            self, indices: Iterable[Mapping[Indices, Union[int, slice]]]
    ) -> Iterable[np.ndarray]:
        """Given an iterable of indices, return a generator of numpy arrays from self

        Parameters
        ----------
        indices, Iterable[Mapping[str, int]]
            Iterable of indices that map a dimension (str) to a value (int)

        Yields
        ------
        np.ndarray
            Numpy array that corresponds to provided indices
        """
        for inds in indices:
            array, axes = self.get_slice(inds)
            yield array

    def apply(self, func, is_volume=False, in_place=True, verbose: bool=False, **kwargs) -> "ImageStack":
        """Apply func over all tiles or volumes in self

        Parameters
        ----------
        func : Callable
            Function to apply. must expect a first argument which is a 2d or 3d numpy array (see is_volume) and return a
            np.ndarray. If inplace is True, must return an array of the same shape.
        is_volume : bool
            (default False) If True, pass 3d volumes (x, y, z) to func
        in_place : bool
            (default True) If True, function is executed in place. If n_proc is not 1, the tile or
            volume will be copied once during execution. If false, a new ImageStack object will be produced.
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        kwargs : dict
            Additional arguments to pass to func

        Returns
        -------
        ImageStack :
            If inplace is False, return a new ImageStack, otherwise return a reference to the original stack with
            data modified by application of func
        """
        if not in_place:
            image_stack = deepcopy(self)
            return image_stack.apply(func, is_volume=is_volume, in_place=True, verbose=verbose, **kwargs)

        mapfunc: Callable = map  # TODO: ambrosejcarr posix-compliant multiprocessing
        indices = list(self._iter_indices(is_volume=is_volume))

        if verbose:
            tiles = tqdm(self._iter_tiles(indices))
        else:
            tiles = self._iter_tiles(indices)

        applyfunc: Callable = partial(func, **kwargs)
        results = mapfunc(applyfunc, tiles)

        for r, inds in zip(results, indices):
            self.set_slice(inds, r)

        return self

    def transform(self, func, is_volume=False, verbose=False, **kwargs) -> List[Any]:
        """Apply func over all tiles or volumes in self

        Parameters
        ----------
        func : Callable
            Function to apply. must expect a first argument which is a 2d or 3d numpy array (see is_volume) but
            may return any object type
        is_volume : bool
            (default False) If True, pass 3d volumes (x, y, z) to func
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        kwargs : dict
            Additional arguments to pass to func being applied

        Returns
        -------
        List[Any] :
            The results of applying func to stored image data
        """
        mapfunc: Callable = map
        indices = list(self._iter_indices(is_volume=is_volume))

        if verbose:
            tiles = tqdm(self._iter_tiles(indices))
        else:
            tiles = self._iter_tiles(indices)

        applyfunc: Callable = partial(func, **kwargs)
        results = mapfunc(applyfunc, tiles)

        return list(zip(results, indices))

    @property
    def tile_metadata(self) -> pd.DataFrame:
        """return a table containing Tile metadata

        Returns
        -------
        pd.DataFrame :
            dataframe containing per-tile metadata information for each image. Guaranteed to include information on
            channel, imaging round, z_layer, and barcode index. Also contains any information stored in the
            extras field for each tile in hybridization.json

        """

        data: collections.defaultdict = collections.defaultdict(list)
        index_keys = set(
            key
            for tile in self._image_partition.tiles()
            for key in tile.indices.keys())
        extras_keys = set(
            key
            for tile in self._image_partition.tiles()
            for key in tile.extras.keys())
        duplicate_keys = index_keys.intersection(extras_keys)
        if len(duplicate_keys) > 0:
            duplicate_keys_str = ", ".join([str(key) for key in duplicate_keys])
            raise ValueError(
                f"keys ({duplicate_keys_str}) was found in both the Tile specification and extras field. Tile "
                f"specification keys may not be duplicated in the extras field.")

        for tile in self._image_partition.tiles():
            for k in index_keys:
                data[k].append(tile.indices.get(k, None))
            for k in extras_keys:
                data[k].append(tile.extras.get(k, None))

            if 'barcode_index' not in tile.extras:
                round_ = tile.indices[Indices.ROUND]
                ch = tile.indices[Indices.CH]
                z = tile.indices.get(Indices.Z, 0)
                barcode_index = (((z * self.num_rounds) + round_) * self.num_chs) + ch

                data['barcode_index'].append(barcode_index)

        return pd.DataFrame(data)

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
        Returns the shape of the space that this image inhabits.  It does not include the dimensions of the image
        itself.  For instance, if this is an X-Y image in a C-H-Y-X space, then the shape would include the dimensions C
        and H.

        Returns
        -------
        An ordered mapping between index names to the size of the index.
        """
        # TODO: (ttung) Note that the return type should be ..OrderedDict[Any, str], but python3.6 has a bug where this
        # breaks horribly.  Can't find a bug id to link to, but see
        # https://stackoverflow.com/questions/41207128/how-do-i-specify-ordereddict-k-v-types-for-mypy-type-annotation
        result: collections.OrderedDict[Any, str] = collections.OrderedDict()
        for name, data in ImageStack.AXES_DATA.items():
            result[name] = self._data.shape[data.order]
        result['y'] = self._data.shape[-2]
        result['x'] = self._data.shape[-1]

        return result

    def _get_dimension_size(self, dimension: Indices):
        axis_data = ImageStack.AXES_DATA[dimension]
        if dimension in self._image_partition.dimensions or axis_data.required:
            return self._image_partition.get_dimension_shape(dimension)
        return 1

    @property
    def num_rounds(self):
        return self._get_dimension_size(Indices.ROUND)

    @property
    def num_chs(self):
        return self._get_dimension_size(Indices.CH)

    @property
    def num_zlayers(self):
        return self._get_dimension_size(Indices.Z)

    @property
    def tile_shape(self):
        return self._tile_shape

    def write(self, filepath: str, tile_opener=None) -> None:
        """write the image tensor to disk

        Parameters
        ----------
        filepath : str
            path + prefix for writing the image tensor
        tile_opener : TODO ttung: doc me.

        """
        if self._data_needs_writeback:
            for tile in self._image_partition.tiles():
                h = tile.indices[Indices.ROUND]
                c = tile.indices[Indices.CH]
                zlayer = tile.indices.get(Indices.Z, 0)
                tile.numpy_array, axes = self.get_slice(indices={Indices.ROUND: h, Indices.CH: c, Indices.Z: zlayer})
                assert len(axes) == 0
            self._data_needs_writeback = False

        seen_x_coords, seen_y_coords, seen_z_coords = set(), set(), set()
        for tile in self._image_partition.tiles():
            seen_x_coords.add(tile.coordinates[Coordinates.X])
            seen_y_coords.add(tile.coordinates[Coordinates.Y])
            z_coords = tile.coordinates.get(Coordinates.Z, None)
            if z_coords is not None:
                seen_z_coords.add(z_coords)

        sorted_x_coords = sorted(seen_x_coords)
        sorted_y_coords = sorted(seen_y_coords)
        sorted_z_coords = sorted(seen_z_coords)
        x_coords_to_idx = {coords: idx for idx, coords in enumerate(sorted_x_coords)}
        y_coords_to_idx = {coords: idx for idx, coords in enumerate(sorted_y_coords)}
        z_coords_to_idx = {coords: idx for idx, coords in enumerate(sorted_z_coords)}

        if tile_opener is None:
            def tile_opener(tileset_path, tile, ext):
                tile_basename = os.path.splitext(tileset_path)[0]
                xcoord = tile.coordinates[Coordinates.X]
                ycoord = tile.coordinates[Coordinates.Y]
                zcoord = tile.coordinates.get(Coordinates.Z, None)
                xcoord = tuple(xcoord) if isinstance(xcoord, list) else xcoord
                ycoord = tuple(ycoord) if isinstance(ycoord, list) else ycoord
                xval = x_coords_to_idx[xcoord]
                yval = y_coords_to_idx[ycoord]
                if zcoord is not None:
                    zval = z_coords_to_idx[zcoord]
                    zstr = "-Z{}".format(zval)
                else:
                    zstr = ""
                return open(
                    "{}-X{}-Y{}{}-H{}-C{}.{}".format(
                        tile_basename,
                        xval,
                        yval,
                        zstr,
                        tile.indices[Indices.ROUND],
                        tile.indices[Indices.CH],
                        ext,
                    ),
                    "wb")

        Writer.write_to_path(
            self._image_partition,
            filepath,
            pretty=True,
            tile_opener=tile_opener)

    def max_proj(self, *dims: Indices) -> np.ndarray:
        """return a max projection over one or more axis of the image tensor

        Parameters
        ----------
        dims : Indices
            one or more axes to project over

        Returns
        -------
        np.ndarray :
            max projection

        """
        axes = list()

        # preserve data's type
        dtype = self.numpy_array.dtype

        for dim in dims:
            try:
                axes.append(ImageStack.AXES_DATA[dim].order)
            except KeyError:
                raise ValueError(
                    "Dimension: {} not supported. Expecting one of: {}".format(dim, ImageStack.AXES_DATA.keys()))

        max_projection = np.max(self._data, axis=tuple(axes)).astype(dtype)
        return max_projection

    @staticmethod
    def _default_tile_extras_provider(round_: int, ch: int, z: int) -> Any:
        """
        Returns None for extras for any given round/ch/z.
        """
        return None

    @staticmethod
    def _default_tile_data_provider(round_: int, ch: int, z: int, height: int, width: int) -> np.ndarray:
        """
        Returns a tile of just ones for any given round/ch/z.
        """
        return np.ones((height, width))

    @classmethod
    def synthetic_stack(
            cls,
            num_round: int=4,
            num_ch: int=4,
            num_z: int=12,
            tile_height: int=50,
            tile_width: int=40,
            tile_data_provider: Callable[[int, int, int, int, int], np.ndarray]=None,
            tile_extras_provider: Callable[[int, int, int], Any]=None,
    ) -> "ImageStack":
        """generate a synthetic ImageStack

        Returns
        -------
        ImageStack :
            imagestack containing a tensor whose default shape is (2, 3, 4, 30, 20)
            and whose default values are all 1.

        """
        if tile_data_provider is None:
            tile_data_provider = cls._default_tile_data_provider
        if tile_extras_provider is None:
            tile_extras_provider = cls._default_tile_extras_provider

        img = TileSet(
            {Coordinates.X, Coordinates.Y, Indices.ROUND, Indices.CH, Indices.Z},
            {
                Indices.ROUND: num_round,
                Indices.CH: num_ch,
                Indices.Z: num_z,
            },
            default_tile_shape=(tile_height, tile_width),
        )
        for round_ in range(num_round):
            for ch in range(num_ch):
                for z in range(num_z):
                    tile = Tile(
                        {
                            Coordinates.X: (0.0, 0.001),
                            Coordinates.Y: (0.0, 0.001),
                            Coordinates.Z: (0.0, 0.001),
                        },
                        {
                            Indices.ROUND: round_,
                            Indices.CH: ch,
                            Indices.Z: z,
                        },
                        extras=tile_extras_provider(round_, ch, z),
                    )
                    tile.numpy_array = tile_data_provider(round_, ch, z, tile_height, tile_width)

                    img.add_tile(tile)

        stack = cls(img)
        return stack

    @classmethod
    def synthetic_spots(
            cls, intensities: IntensityTable, num_z: int, height: int, width: int,
            n_photons_background=1000, point_spread_function=(4, 2, 2),
            camera_detection_efficiency=0.25, background_electrons=1,
            graylevel: float=37000.0 / 2 ** 16, ad_conversion_bits=16,
    ) -> "ImageStack":
        """Generate a synthetic ImageStack from a set of Features stored in an IntensityTable

        Parameters
        ----------
        intensities : IntensityTable
            IntensityTable containing coordinates of fluorophores. Used to position and generate
            spots in the output ImageStack
        num_z : int
            Number of z-planes in the ImageStack
        height : int
            Height in pixels of the ImageStack
        width : int
            Width in pixels of the ImageStack
        n_photons_background : int
            Poisson rate for the number of background photons to add to each pixel of the image.
            Set this parameter to 0 to eliminate background.
            (default 1000)
        point_spread_function : Tuple[int]
            The width of the gaussian density wherein photons spread around their light source.
            Set to zero to eliminate this (default (4, 2, 2))
        camera_detection_efficiency : float
            The efficiency of the camera to detect light. Set to 1 to remove this filter (default
            0.25)
        background_electrons : int
            Poisson rate for the number of spurious electrons detected per pixel during image
            capture by the camera (default 1)
        graylevel : float
            The number of shades of gray displayable by the synthetic camera. Larger numbers will
            produce higher resolution images (default 37000 / 2 ** 16)
        ad_conversion_bits : int
            The number of bits used during analog to visual conversion (default 16)

        Returns
        -------

        """
        # check some params
        if not 0 < camera_detection_efficiency <= 1:
            raise ValueError(
                f'invalid camera_detection_efficiency value: {camera_detection_efficiency}. '
                f'Must be in the interval (0, 1].')

        def select_uint_dtype(array):
            """choose appropriate dtype based on values of an array"""
            max_val = np.max(array)
            for dtype in (np.uint8, np.uint16, np.uint32):
                if max_val <= np.iinfo(dtype).max:
                    return array.astype(dtype)
            raise ValueError('value exceeds dynamic range of largest skimage-supported type')

        # make sure requested dimensions are large enough to support intensity values
        indices = zip((Features.Z, Features.Y, Features.X), (num_z, height, width))
        for index, requested_size in indices:
            required_size = intensities.coords[index].values.max()
            if required_size > requested_size:
                raise ValueError(
                    f'locations of intensities contained in table exceed the size of requested '
                    f'dimension {index}. Required size {required_size} > {requested_size}.')

        # create an empty array of the correct size
        image = np.zeros(
            (
                intensities.sizes[Indices.ROUND.value],
                intensities.sizes[Indices.CH.value],
                num_z,
                height,
                width
            )
        )

        for ch, round_ in product(*(range(s) for s in intensities.shape[1:])):
            spots = intensities[:, ch, round_]

            # numpy deprecated casting a specific way of casting floats that is triggered in xarray
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                values = spots.where(spots, drop=True)

            image[round_, ch, values.z, values.y, values.x] = values

        # add imaging noise
        image += np.random.poisson(n_photons_background, size=image.shape)

        # blur image over coordinates, but not over round_/channels (dim 0, 1)
        sigma = (0, 0) + point_spread_function
        image = gaussian_filter(image, sigma=sigma, mode='nearest')

        image *= camera_detection_efficiency

        image += np.random.normal(scale=background_electrons, size=image.shape)

        # mimic analog to digital conversion
        image = (image / graylevel).astype(int).clip(0, 2 ** ad_conversion_bits)

        # clip in case we've picked up some negative values
        image = np.clip(image, 0, a_max=None)

        # set correct dtype and convert to stack
        image = select_uint_dtype(image)
        return cls.from_numpy_array(image)

    def squeeze(self) -> np.ndarray:
        """return an array that is linear over categorical dimensions and z

        Returns
        -------
        np.ndarray :
            array of shape (num_rounds + num_channels + num_z_layers, x, y).

        """
        first_dim = self.num_rounds * self.num_chs * self.num_zlayers
        new_shape = (first_dim,) + self.tile_shape
        new_data = self.numpy_array.reshape(new_shape)

        return new_data

    def un_squeeze(self, stack):
        if type(stack) is list:
            stack = np.array(stack)

        new_shape = (self.num_rounds, self.num_chs, self.num_zlayers) + self.tile_shape
        res = stack.reshape(new_shape)
        return res

    @classmethod
    def from_numpy_array(cls, array: np.ndarray) -> "ImageStack":
        """Create an ImageStack from a 5d numpy array with shape (n_round, n_ch, n_z, y, x)

        Parameters
        ----------
        array : np.ndarray
            5-d tensor of shape (n_round, n_ch, n_z, y, x)

        Returns
        -------
        ImageStack :
            array data stored as an ImageStack

        """
        if len(array.shape) != 5:
            raise ValueError('a 5-d tensor with shape (n_round, n_ch, n_z, y, x) must be provided.')
        n_round, n_ch, n_z, height, width = array.shape
        empty = cls.synthetic_stack(
            num_round=n_round, num_ch=n_ch, num_z=n_z, tile_height=height, tile_width=width)

        # preserve original dtype
        empty._data = empty._data.astype(array.dtype)

        for h in np.arange(n_round):
            for c in np.arange(n_ch):
                for z in np.arange(n_z):
                    view = array[h, c, z]
                    empty.set_slice({Indices.ROUND: h, Indices.CH: c, Indices.Z: z}, view)

        return empty
