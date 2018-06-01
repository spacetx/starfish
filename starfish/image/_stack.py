import collections
import os
from functools import partial
from typing import Any, Iterable, Iterator, Mapping, MutableSequence, Sequence, Tuple, Union
from warnings import warn

import numpy
from scipy.stats import scoreatpercentile
from skimage import exposure
from slicedimage import Reader, Writer

from starfish.constants import Coordinates, Indices
from starfish.errors import DataFormatWarning
from ._base import ImageBase


class ImageStack(ImageBase):
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
        self._tile_shape = tuple(image_partition.default_tile_shape)

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
                # this source tile has a smaller data size than the other ones, though the same kind.  need to scale the
                # data.
                if data.dtype.kind == "f":
                    # floating point can be done with numpy.interp.
                    src_finfo = numpy.finfo(data.dtype)
                    dst_finfo = numpy.finfo(self._data.dtype)
                    data = numpy.interp(
                        data,
                        (src_finfo.min, src_finfo.max),
                        (dst_finfo.min, dst_finfo.max))
                else:
                    # fixed point can be done with a simple multiply.
                    src_max = numpy.iinfo(data.dtype).max
                    dst_max = numpy.iinfo(self._data.dtype).max
                    data = data * (dst_max / src_max)
                warn(
                    f"Tile "
                    f"(H: {tile.indices[Indices.HYB]} C: {tile.indices[Indices.CH]} Z: {tile.indices[Indices.Z]}) has "
                    f"dtype {data.dtype}.  One or more tiles is of a larger dtype {self._data.dtype}.",
                    DataFormatWarning)
            self.set_slice(indices={Indices.HYB: h, Indices.CH: c, Indices.Z: zlayer}, data=data)
        # set_slice will mark the data as needing writeback, so we need to unset that.
        self._data_needs_writeback = False

    @classmethod
    def from_url(cls, relativeurl, baseurl):
        image_partition = Reader.parse_doc(relativeurl, baseurl)

        return cls(image_partition)

    @property
    def numpy_array(self):
        result = self._data.view()
        result.setflags(write=False)
        return result

    @numpy_array.setter
    def numpy_array(self, data):
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
            color_map: str= 'gray', rescale: bool=False, figure_size: Tuple[int, int]=(10, 10)):
        """Create an interactive visualization of an image stack

        Produces a slider that flips through the selected volume tile-by-tile

        Parameters
        ----------
        indices : Mapping[Indices, Union[int, slice]],
            Indices to select a volume to visualize. Passed to `Image.get_slice()`.
            See `Image.get_slice()` for examples.
        color_map : str (default = 'gray')
            string id of a matplotlib colormap
        rescale : bool (default = True)
            if True, rescale the data to exclude high and low-value outliers (see skimage.exposure.rescale_intensity)
        figure_size : Tuple[int, int] (default = (10, 10))
            size of the figure in inches

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
        linear_view = data.reshape((n,) + data.shape[-2:])

        # set the labels for the linearized tiles
        from itertools import product
        labels = []
        for index, size in zip(remaining_inds, data.shape[:-2]):
            labels.append([f'{index}{n}' for n in range(size)])
        labels = list(product(*labels))

        n = linear_view.shape[0]

        if rescale:
            print("Rescaling ...")
            vmin, vmax = scoreatpercentile(data, (0.5, 99.5))
            linear_view = exposure.rescale_intensity(
                linear_view,
                in_range=(vmin, vmax),
                out_range=numpy.float32
            ).astype(numpy.float32)

        def show_plane(ax, plane, cmap="gray", title=None):
            ax.imshow(plane, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])

            if title:
                ax.set_title(title)

        @interact(plane=(0, n - 1))
        def display_slice(plane=34):
            fig, ax = plt.subplots(figsize=figure_size)
            show_plane(ax, linear_view[plane], title=f'{labels[plane]}', cmap=color_map)
            plt.show()

        return display_slice

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
        """Given an iterable of indices, return a generator of numpy arrays from self.image

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

    def apply(self, func, is_volume=False, **kwargs):
        """Apply func over all tiles or volumes in self

        Parameters
        ----------
        func : Callable
            Function to apply. must expect a first argument which is a 2d or 3d numpy array (see is_volume) and return a
            numpy.ndarray. If inplace is True, must return an array of the same shape.
        is_volume : bool
            (default False) If True, pass 3d volumes (x, y, z) to func
        inplace : bool
            (default True) If True, function is executed in place. If n_proc is not 1, the tile or
            volume will be copied once during execution. Not currently implemented.
        kwargs : dict
            Additional arguments to pass to func

        Returns
        -------
        Optional[ImageStack]
            If inplace is False, return a new ImageStack containing the output of apply
        """
        mapfunc = map  # TODO: ambrosejcarr posix-compliant multiprocessing
        indices = list(self._iter_indices(is_volume=is_volume))
        tiles = self._iter_tiles(indices)

        applyfunc = partial(func, **kwargs)

        results = mapfunc(applyfunc, tiles)

        for r, inds in zip(results, indices):
            self.set_slice(inds, r)

        # TODO: ambrosejcarr implement inplace=False

    @property
    def raw_shape(self) -> Tuple[int]:
        return self._data.shape

    @property
    def shape(self) -> collections.OrderedDict:
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

    def write(self, filepath, tile_opener=None):
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

    def max_proj(self, *dims):
        axes = list()
        for dim in dims:
            try:
                axes.append(ImageStack.AXES_MAP[dim])
            except KeyError:
                raise ValueError(
                    "Dimension: {} not supported. Expecting one of: {}".format(dim, ImageStack.AXES_MAP.keys()))

        return numpy.max(self._data, axis=tuple(axes))
