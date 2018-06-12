import collections
import os
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableSequence, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy
import pandas as pd
from scipy.stats import scoreatpercentile
from skimage import exposure
from slicedimage import Reader, Writer
from slicedimage.io import resolve_path_or_url
from tqdm import tqdm

from starfish.constants import Coordinates, Indices
from starfish.errors import DataFormatWarning
from starfish.pipeline.features.spot_attributes import SpotAttributes


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
    num_hybs     the number of hybridization rounds stored in the image tensor
    num_zlayers  the number of z-layers stored in the image tensor
    numpy_array  the 5-d image tensor is stored in this array
    raw_shape    the shape of the image tensor (in integers)
    shape        the shape of the image tensor by categorical index (channels, hybridization rounds, z-layers)
    """

    AXES_MAP = {
        Indices.HYB: 0,
        Indices.CH: 1,
        Indices.Z: 2,
    }
    N_AXES = max(AXES_MAP.values()) + 1

    def __init__(self, image_partition):
        self._image_partition = image_partition
        self._num_hybs = image_partition.get_dimension_shape(Indices.HYB)
        self._num_chs = image_partition.get_dimension_shape(Indices.CH)
        if Indices.Z in image_partition.dimensions:
            self._num_zlayers = image_partition.get_dimension_shape(Indices.Z)
        else:
            self._num_zlayers = 1
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

        # now that we know the tile data type (kind and size), we can allocate the data array.
        self._data = numpy.zeros(
            shape=(self._num_hybs, self._num_chs, self._num_zlayers) + self._tile_shape,
            dtype=numpy.dtype(f"{kind}{max_size}")
        )

        # iterate through the tiles and set the data.
        for tile in self._image_partition.tiles():
            h = tile.indices[Indices.HYB]
            c = tile.indices[Indices.CH]
            zlayer = tile.indices.get(Indices.Z, 0)
            data = tile.numpy_array
            if max_size != data.dtype.itemsize:
                if data.dtype.kind == "i" or data.dtype.kind == "u":
                    # fixed point can be done with a simple multiply.
                    src_range = numpy.iinfo(data.dtype).max - numpy.iinfo(data.dtype).min + 1
                    dst_range = numpy.iinfo(self._data.dtype).max - numpy.iinfo(self._data.dtype).min + 1
                    data = data * (dst_range / src_range)
                warn(
                    f"Tile "
                    f"(H: {tile.indices[Indices.HYB]} C: {tile.indices[Indices.CH]} Z: {tile.indices[Indices.Z]}) has "
                    f"dtype {data.dtype}.  One or more tiles is of a larger dtype {self._data.dtype}.",
                    DataFormatWarning)
            self.set_slice(indices={Indices.HYB: h, Indices.CH: c, Indices.Z: zlayer}, data=data)
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
    ) -> Tuple[numpy.ndarray, Sequence[Indices]]:
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
            data: numpy.ndarray,
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
                numpy.moveaxis(data, move_src, move_dst)

        if self._data[slice_list].shape != data.shape:
            raise ValueError("source shape {} mismatches destination shape {}".format(
                data.shape, self._data[slice_list].shape))

        self._data[slice_list] = data
        self._data_needs_writeback = True

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
        n = numpy.dot(*data.shape[:-2])

        # linearize the array
        linear_view: numpy.ndarray = data.reshape((n,) + data.shape[-2:])

        # set the labels for the linearized tiles

        labels = []
        for index, size in zip(remaining_inds, data.shape[:-2]):
            labels.append([f'{index}{n}' for n in range(size)])
        labels = list(product(*labels))

        n = linear_view.shape[0]

        if rescale and any((p_min, p_max)):
            raise ValueError('select one of rescale and p_min/p_max to rescale image, not both.')

        elif rescale is not None:
            print("Rescaling ...")
            vmin, vmax = scoreatpercentile(data, (0.5, 99.5))
            linear_view = exposure.rescale_intensity(
                linear_view,
                in_range=(vmin, vmax),
                out_range=numpy.float32
            ).astype(numpy.float32)

        elif p_min or p_max:
            print("Clipping ...")
            a_min, a_max = scoreatpercentile(
                linear_view,
                (p_min if p_min else 0, p_max if p_max else 100)
            )
            linear_view = numpy.clip(linear_view, a_min=a_min, a_max=a_max)

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

        if z is not None and z in result_df.columns:
            inds = numpy.abs(result_df['z'] - z) < z_dist
        else:
            inds = numpy.ones(result_df.shape[0]).astype(bool)

        # get the data needed to plot
        selected = result_df.loc[inds, ['r', 'x', 'y']]

        for i in numpy.arange(selected.shape[0]):
            r, x, y = selected.iloc[i, :]  # radius is a duplicate, and is present twice
            c = plt.Circle((x, y), r * scale_radius, color='r', linewidth=size, fill=False)
            ax.add_patch(c)

    def _build_slice_list(
            self,
            indices: Mapping[Indices, Union[int, slice]]
    ) -> Tuple[Tuple[Union[int, slice], ...], Sequence[Indices]]:
        slice_list: MutableSequence[Union[int, slice]] = [
            slice(None, None, None)
            for _ in range(ImageStack.N_AXES)
        ]
        axes = []
        removed_axes = set()
        for name, value in indices.items():
            idx = ImageStack.AXES_MAP[name]
            if not isinstance(value, slice):
                removed_axes.add(name)
            slice_list[idx] = value

        for dimension_value, dimension_name in sorted([
            (dimension_value, dimension_name)
            for dimension_name, dimension_value in ImageStack.AXES_MAP.items()
        ]):
            if dimension_name not in removed_axes:
                axes.append(dimension_name)

        return tuple(slice_list), axes

    def _iter_indices(self, is_volume: bool=False) -> Iterator[Mapping[Indices, int]]:
        """Iterate over indices of image tiles or image volumes if is_volume is True

        Parameters
        ----------
        is_volume, bool
            If True, yield indices necessary to extract volumes from self, else return
            indices for tiles

        Yields
        ------
        Dict[str, int]
            Mapping of dimension name to index

        """
        for hyb in numpy.arange(self.shape[Indices.HYB]):
            for ch in numpy.arange(self.shape[Indices.CH]):
                if is_volume:
                    yield {Indices.HYB: hyb, Indices.CH: ch}
                else:
                    for z in numpy.arange(self.shape[Indices.Z]):
                        yield {Indices.HYB: hyb, Indices.CH: ch, Indices.Z: z}

    def _iter_tiles(
            self, indices: Iterable[Mapping[Indices, Union[int, slice]]]
    ) -> Iterable[numpy.ndarray]:
        """Given an iterable of indices, return a generator of numpy arrays from self

        Parameters
        ----------
        indices, Iterable[Mapping[str, int]]
            Iterable of indices that map a dimension (str) to a value (int)

        Yields
        ------
        numpy.ndarray
            Numpy array that corresponds to provided indices
        """
        for inds in indices:
            array, axes = self.get_slice(inds)
            yield array

    def apply(self, func, is_volume=False, in_place=True, verbose: bool=False, **kwargs):
        """Apply func over all tiles or volumes in self

        Parameters
        ----------
        func : Callable
            Function to apply. must expect a first argument which is a 2d or 3d numpy array (see is_volume) and return a
            numpy.ndarray. If inplace is True, must return an array of the same shape.
        is_volume : bool
            (default False) If True, pass 3d volumes (x, y, z) to func
        in_place : bool
            (default True) If True, function is executed in place. If n_proc is not 1, the tile or
            volume will be copied once during execution. If false, the outputs of the function executed on individual
            tiles or volumes will be output as a list
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        kwargs : dict
            Additional arguments to pass to func

        Returns
        -------
        Optional[List[Tuple[np.ndarray, Mapping[Indices: Union[int, slice]]]]
            If inplace is False, return the results of applying func to stored image data
        """
        mapfunc: Callable = map  # TODO: ambrosejcarr posix-compliant multiprocessing
        indices = list(self._iter_indices(is_volume=is_volume))

        if verbose:
            tiles = tqdm(self._iter_tiles(indices))
        else:
            tiles = self._iter_tiles(indices)

        applyfunc: Callable = partial(func, **kwargs)
        results = mapfunc(applyfunc, tiles)

        # TODO ttung: this should return an ImageStack, not a bunch of indices.
        if not in_place:
            return list(zip(results, indices))

        for r, inds in zip(results, indices):
            self.set_slice(inds, r)

    @property
    def tile_metadata(self) -> pd.DataFrame:
        """return a table containing Tile metadata

        Returns
        -------
        pd.DataFrame :
            dataframe containing per-tile metadata information for each image. Guaranteed to include information on
            channel, hybridization round, z_layer, and barcode index. Also contains any information stored in the
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
                hyb = tile.indices[Indices.HYB]
                ch = tile.indices[Indices.CH]
                z = tile.indices.get(Indices.Z, 0)
                barcode_index = (((z * self.num_hybs) + hyb) * self.num_chs) + ch

                data['barcode_index'].append(barcode_index)

        return pd.DataFrame(data)

    @property
    def raw_shape(self) -> Tuple[int]:
        """
        Returns the shape of the space that this image inhabits.  It does not include the dimensions of the image
        itself.  For instance, if this is an X-Y image in a C-H-Y-X space, then the shape would include the dimensions C
        and H.

        Returns
        -------
        Tuple[int] :
            The sizes of the indices.
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
        for name, idx in ImageStack.AXES_MAP.items():
            result[name] = self._data.shape[idx]
        result['y'] = self._data.shape[-2]
        result['x'] = self._data.shape[-1]

        return result

    @property
    def num_hybs(self):
        return self._num_hybs

    @property
    def num_chs(self):
        return self._num_chs

    @property
    def num_zlayers(self):
        return self._num_zlayers

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
                h = tile.indices[Indices.HYB]
                c = tile.indices[Indices.CH]
                zlayer = tile.indices.get(Indices.Z, 0)
                tile.numpy_array, axes = self.get_slice(indices={Indices.HYB: h, Indices.CH: c, Indices.Z: zlayer})
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
                        tile.indices[Indices.HYB],
                        tile.indices[Indices.CH],
                        ext,
                    ),
                    "wb")

        Writer.write_to_path(
            self._image_partition,
            filepath,
            pretty=True,
            tile_opener=tile_opener)

    def max_proj(self, *dims: Indices) -> numpy.ndarray:
        """return a max projection over one or more axis of the image tensor

        Parameters
        ----------
        dims : Indices
            one or more axes to project over

        Returns
        -------
        numpy.ndarray :
            max projection

        """
        axes = list()
        for dim in dims:
            try:
                axes.append(ImageStack.AXES_MAP[dim])
            except KeyError:
                raise ValueError(
                    "Dimension: {} not supported. Expecting one of: {}".format(dim, ImageStack.AXES_MAP.keys()))

        return numpy.max(self._data, axis=tuple(axes))

    def squeeze(self) -> numpy.ndarray:
        """return an array that is linear over categorical dimensions and z

        Returns
        -------
        np.ndarray :
            array of shape (num_hybs + num_channels + num_z_layers, x, y).

        """
        first_dim = self.num_hybs * self.num_chs * self.num_zlayers
        new_shape = (first_dim,) + self.tile_shape
        new_data = self.numpy_array.reshape(new_shape)

        return new_data

    def un_squeeze(self, stack):
        if type(stack) is list:
            stack = numpy.array(stack)

        new_shape = (self.num_hybs, self.num_chs, self.num_zlayers) + self.tile_shape
        res = stack.reshape(new_shape)
        return res
