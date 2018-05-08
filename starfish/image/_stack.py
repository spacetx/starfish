import os

import numpy
from slicedimage import Reader, Writer

from ._base import ImageBase
from ._constants import Coordinates, Indices


class ImageStack(ImageBase):
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
            self._data[h, c, zlayer, :] = tile.numpy_array

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

    @property
    def shape(self):
        if self._data is None:
            return None
        else:
            return self._data.shape

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
                tile.numpy_array = self._data[h, c, zlayer, :]
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
            def tile_opener(toc_path, tile, ext):
                tile_basename = os.path.splitext(toc_path)[0]
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
        valid_dims = {
            Indices.HYB: 0,
            Indices.CH: 1,
            Indices.Z: 2,
        }
        axes = list()
        for dim in dims:
            try:
                axes.append(valid_dims[dim])
            except KeyError:
                raise ValueError("Dimension: {} not supported. Expecting one of: {}".format(dim, valid_dims.keys()))

        return numpy.max(self._data, axis=tuple(axes))
