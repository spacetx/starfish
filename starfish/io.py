import json
import os

import numpy as np
import pandas as pd

from .image import ImageFormat, ImageStack
from .munge import list_to_stack


class Stack:
    def __init__(self):
        # data organization
        self.org = None
        self.path = None
        self.image = None

        # auxilary images
        self.aux_dict = dict()

        # readers and writers
        self.read_fn = None  # set by self._read_metadata
        self.write_fn = np.save  # asserted for now

    def read(self, in_json):
        # TODO: (ttung) remove this hackery
        self.path = os.path.dirname(in_json)
        with open(in_json, 'r') as in_file:
            self.org = json.load(in_file)
        self.image = ImageStack.from_org_json(in_json)
        self._read_aux()

    def _read_aux(self):
        data_dicts = self.org['aux']

        for d in data_dicts:
            typ = d['type']
            fname = d['file']
            img_format = ImageFormat[d['format']]
            self.aux_dict[typ] = img_format.reader_func(os.path.join(self.path, fname))

    # TODO should this thing write npy?
    def write(self, dir_name):
        self._write_stack(dir_name)
        self._write_aux(dir_name)
        self._write_metadata(dir_name)

    def _write_metadata(self, dir_name):
        self.org['metadata']['format'] = ImageFormat.NUMPY.name

        with open(os.path.join(dir_name, 'org.json'), 'w') as outfile:
            json.dump(self.org, outfile, indent=4)

    def _write_stack(self, dir_name):
        self.org['data'] = self.image.write(os.path.join(dir_name, "org.json"))

    def _write_aux(self, dir_name):
        for d in self.org['aux']:
            typ = d['type']
            fname = os.path.splitext(d['file'])[0]
            d['file'] = "{}.{}".format(fname, ImageFormat.NUMPY.file_ext)
            d['format'] = ImageFormat.NUMPY.name
            self.write_fn(os.path.join(dir_name, fname), self.aux_dict[typ])

    def set_stack(self, new_stack):
        if new_stack.shape != self.image.shape:
            msg = "Shape mismatch. Current data shape: {}, new data shape: {}".format(
                self.image.shape, new_stack.shape)
            raise AttributeError(msg)
        self.image = ImageStack(
            new_stack, ImageFormat.NUMPY.name, self.image.num_hybs, self.image.num_chs, self.image.tile_shape)

    def set_aux(self, key, img):
        if key in self.aux_dict:
            old_img = self.aux_dict[key]
            if old_img.shape != img.shape:
                msg = "Shape mismatch. Current data shape: {}, new data shape: {}".format(
                    old_img.shape, img.shape)
                raise AttributeError(msg)
        else:
            self.org['aux'].append({'file': key, 'type': key, 'format': ImageFormat.NUMPY.name})
        self.aux_dict[key] = img

    def max_proj(self, dim):
        return self.image.max_proj(dim)

    def squeeze(self, bit_map_flag=False):
        first_dim = self.image.num_hybs * self.image.num_chs

        new_shape = (first_dim,) + self.image.tile_shape
        new_data = self.image.numpy_array.reshape(new_shape)

        ind = np.arange(first_dim)

        # e.g. 0, 1, 2, 3 --> 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
        hyb = np.repeat(np.arange(self.image.num_hybs), self.image.num_chs)

        # e.g. 0, 1, 2, 3 --> 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
        ch = np.tile(np.arange(self.image.num_chs), self.image.num_hybs)
        self.squeeze_map = pd.DataFrame(dict(ind=ind, hyb=hyb, ch=ch))

        if bit_map_flag:
            mp = [(d['hyb'], d['ch'], d['bit']) for d in self.org['data']]
            mp = pd.DataFrame(mp, columns=['hyb', 'ch', 'bit'])
            self.squeeze_map = pd.merge(self.squeeze_map, mp, on=['ch', 'hyb'], how='left')

        return new_data

    def un_squeeze(self, stack):
        if type(stack) is list:
            stack = list_to_stack(stack)

        new_shape = (self.image.num_hybs, self.image.num_chs) + self.image.tile_shape
        res = stack.reshape(new_shape)
        return res
