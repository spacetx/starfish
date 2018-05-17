import collections
import os
import typing

import numpy
from slicedimage import Reader, Writer

from ._base import ImageBase
from ._constants import Coordinates, Indices


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

        self._data = numpy.zeros((self._num_hybs, self._num_chs, self._num_zlayers) + self._tile_shape)
        self._data_needs_writeback = False

        for tile in image_partition.tiles():
            h = tile.indices[Indices.HYB]
            c = tile.indices[Indices.CH]
            zlayer = tile.indices.get(Indices.Z, 0)
            self.set_slice(indices={Indices.HYB: h, Indices.CH: c, Indices.Z: zlayer}, data=tile.numpy_array)

    @classmethod
    def from_image_stack(cls, image_stack_name_or_url, baseurl):
        image_partition = Reader.parse_doc(image_stack_name_or_url, baseurl)

        return ImageStack(image_partition)

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
            indices: typing.Mapping[Indices, typing.Union[int, slice]]
    ) -> typing.Tuple[numpy.ndarray, typing.Sequence[Indices]]:
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
            indices: typing.Mapping[Indices, typing.Union[int, slice]],
            data: numpy.ndarray,
            axes: typing.Sequence[Indices]=None):
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

    def _build_slice_list(
            self,
            indices: typing.Mapping[Indices, typing.Union[int, slice]]
    ) -> typing.Tuple[typing.Tuple[typing.Union[int, slice], ...], typing.Sequence[Indices]]:
        slice_list = [
            slice(None, None, None)
            for _ in range(ImageStack.N_AXES)
        ]  # type: typing.MutableSequence[typing.Union[int, slice]]
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

    @property
    def raw_shape(self) -> typing.Optional[list]:
        if self._data is None:
            return None

        return self._data.shape

    @property
    def shape(self) -> typing.Optional[dict]:
        if self._data is None:
            return None

        result = collections.OrderedDict()
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
