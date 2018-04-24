import json
import os

import numpy as np
import pandas as pd
from slicedimage import ImageFormat
from slicedimage.io import resolve_url

from .image import ImageStack
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
        self.write_fn = np.save  # asserted for now

    def read(self, in_json):
        # TODO: (ttung) remove this hackery
        self.path = os.path.dirname(in_json)
        with open(in_json, 'r') as in_file:
            self.org = json.load(in_file)

        image_stack_name_or_url = self.org['hybridization_images']

        self.image = ImageStack.from_image_stack(image_stack_name_or_url, "file://{}".format(self.path))
        self._read_aux()

    def _read_aux(self):
        for aux_key, aux_data in self.org['auxiliary_images'].items():
            name_or_url = aux_data['file']
            img_format = ImageFormat[aux_data['tile_format']]
            backend, name, _ = resolve_url(name_or_url, "file://{}".format(self.path))
            with backend.read_file_handle(name) as fh:
                self.aux_dict[aux_key] = img_format.reader_func(fh)

    # TODO should this thing write npy?
    def write(self, dir_name):
        self._write_stack(dir_name)
        self._write_aux(dir_name)
        self._write_metadata(dir_name)

    def _write_metadata(self, dir_name):
        with open(os.path.join(dir_name, 'experiment.json'), 'w') as outfile:
            json.dump(self.org, outfile, indent=4)

    def _write_stack(self, dir_name):
        stack_path = os.path.join(dir_name, "hybridization.json")
        self.image.write(stack_path)
        self.org['hybridization_images'] = stack_path

    def _write_aux(self, dir_name):
        for aux_key, aux_data in self.org['auxiliary_images'].items():
            fname = os.path.splitext(aux_data['file'])[0]
            aux_data['file'] = "{}.{}".format(fname, ImageFormat.NUMPY.file_ext)
            aux_data['tile_format'] = ImageFormat.NUMPY.name
            self.write_fn(os.path.join(dir_name, fname), self.aux_dict[aux_key])

    def set_stack(self, new_stack):
        if new_stack.shape != self.image.shape:
            msg = "Shape mismatch. Current data shape: {}, new data shape: {}".format(
                self.image.shape, new_stack.shape)
            raise AttributeError(msg)
        self.image.numpy_array = new_stack

    def set_aux(self, key, img):
        if key in self.aux_dict:
            old_img = self.aux_dict[key]
            if old_img.shape != img.shape:
                msg = "Shape mismatch. Current data shape: {}, new data shape: {}".format(
                    old_img.shape, img.shape)
                raise AttributeError(msg)
        else:
            self.org['auxiliary_images'][key] = {
                'file': key,
                'tile_format': ImageFormat.NUMPY.name,
                'tile_shape': img.shape,
            }
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
