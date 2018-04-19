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

    @classmethod
    def from_org_json(cls, org_json_path):
        with open(org_json_path, 'r') as in_file:
            org_json = json.load(in_file)
        metadata_dict = org_json['metadata']
        num_hybs = metadata_dict['num_hybs']
        num_chs = metadata_dict['num_chs']
        tile_shape = tuple(metadata_dict['shape'])

        data = numpy.zeros((num_hybs, num_chs) + tile_shape)
        tile_format = ImageFormat[metadata_dict['format']]

        data_dicts = org_json['data']
        base_path = os.path.dirname(org_json_path)

        for data_dict in data_dicts:
            h = data_dict['hyb']
            c = data_dict['ch']
            fname = data_dict['file']
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

    def write(self, filepath, tile_filename_formatter=None):
        prefix = os.path.splitext(os.path.basename(filepath))[0]
        basepath = os.path.dirname(filepath)
        if tile_filename_formatter is None:
            def tile_filename_formatter(x, y, hyb, ch):
                return "{}-x_{}-y_{}-h_{}-c_{}".format(prefix, x, y, hyb, ch)

        data = list()

        for hyb in range(self._num_hybs):
            for ch in range(self._num_chs):
                tile_filename = tile_filename_formatter(0, 0, hyb, ch)
                tile = {
                    'hyb': hyb,
                    'ch': ch,
                    'file': "{}.{}".format(tile_filename, ImageFormat.NUMPY.file_ext),
                }
                numpy.save(os.path.join(basepath, tile_filename), self._data[hyb, ch, :])
                data.append(tile)

        return data

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
            res = self._data

        return res
