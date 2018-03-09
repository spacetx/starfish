import json
import os

import numpy

from ._base import ImageBase, ImageFormat


class ImageStack(ImageBase):
    def __init__(self, data, tile_format, num_hybs, num_chs, tile_shape):
        self._data = data
        self._tile_format = tile_format

        # shape data
        self._num_hybs = num_hybs
        self._num_chs = num_chs
        self._tile_shape = tile_shape

        if len(tile_shape) == 2:
            self._is_volume = False
        else:
            self._is_volume = True

    @classmethod
    def from_image_stack(cls, image_stack_json_path):
        base_path = os.path.dirname(image_stack_json_path)

        with open(image_stack_json_path, 'r') as in_file:
            image_stack = json.load(in_file)

        num_hybs = len(set(tile['coordinates']['hyb'] for tile in image_stack['tiles']))
        num_chs = len(set(tile['coordinates']['ch'] for tile in image_stack['tiles']))
        tile_shape = tuple(image_stack['legend']['default_tile_shape'])
        tile_format = ImageFormat[image_stack['legend']['default_tile_format']]
        data = numpy.zeros((num_hybs, num_chs) + tile_shape)

        for tile_dict in image_stack['tiles']:
            h = tile_dict['coordinates']['hyb']
            c = tile_dict['coordinates']['ch']
            fname = tile_dict['file']
            im = tile_format.reader_func(os.path.join(base_path, fname))
            data[h, c, :] = im

        return ImageStack(data, tile_format, num_hybs, num_chs, tile_shape)

    @property
    def numpy_array(self):
        return self._data

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
    def tile_shape(self):
        return self._tile_shape

    @property
    def is_volume(self):
        return self._is_volume

    def write(self, filepath, tile_filename_formatter=None):
        prefix = os.path.splitext(os.path.basename(filepath))[0]
        basepath = os.path.dirname(filepath)
        if tile_filename_formatter is None:
            def tile_filename_formatter(x, y, hyb, ch):
                return "{}-x_{}-y_{}-h_{}-c_{}".format(prefix, x, y, hyb, ch)

        image_stack = {
            'version': "0.0.0",
            'legend': {
                'dimensions': ["x", "y", "hyb", "ch"],
                'default_tile_shape': self._data[0, 0, :].shape,
                'default_tile_format': ImageFormat.NUMPY.name,
            },
            'tiles': [],
        }

        for hyb in range(self._num_hybs):
            for ch in range(self._num_chs):
                tile_filename = tile_filename_formatter(0, 0, hyb, ch)
                tile = {
                    'coordinates': {
                        'x': 0,
                        'y': 0,
                        'hyb': hyb,
                        'ch': ch,
                    },
                    'file': "{}.{}".format(tile_filename, ImageFormat.NUMPY.file_ext),
                }
                numpy.save(os.path.join(basepath, tile_filename), self._data[hyb, ch, :])
                image_stack['tiles'].append(tile)

        with open(filepath, "w") as fh:
            fh.write(json.dumps(image_stack))

    def max_proj(self, dim):
        valid_dims = ['hyb', 'ch', 'z']
        if dim not in valid_dims:
            msg = "Dimension: {} not supported. Expecting one of: {}".format(dim, valid_dims)
            raise ValueError(msg)

        if dim == 'hyb':
            res = numpy.max(self._data, axis=0)
        elif dim == 'ch':
            res = numpy.max(self._data, axis=1)
        elif dim == 'z' and len(self._tile_shape) > 2:
            res = numpy.max(self._data, axis=4)
        else:
            res = self.data

        return res
