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
        self.image = ImageStack.from_image_stack(os.path.join(self.path, self.org['hybridization']))
        self._read_aux()

    def _read_aux(self):
        for aux_key, aux_data in self.org['aux_img'].items():
            fname = aux_data['file']
            img_format = ImageFormat[aux_data['tile_format']]
            self.aux_dict[aux_key] = img_format.reader_func(os.path.join(self.path, fname))

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
        self.org['hybridization'] = stack_path

    def _write_aux(self, dir_name):
        for aux_key, aux_data in self.org['aux_img'].items():
            fname = os.path.splitext(aux_data['file'])[0]
            aux_data['file'] = "{}.{}".format(fname, ImageFormat.NUMPY.file_ext)
            aux_data['tile_format'] = ImageFormat.NUMPY.name
            self.write_fn(os.path.join(dir_name, fname), self.aux_dict[aux_key])

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
            self.org['aux_img'][key] = {
                'file': key,
                'tile_format': ImageFormat.NUMPY.name,
                'tile_shape': img.shape,
            }
        self.aux_dict[key] = img

    def max_proj(self, dim):
        return self.image.max_proj(dim)

    def squeeze(self, bit_map_flag=False):
        new_shape = ((self.image.num_hybs * self.image.num_chs),) + self.image.tile_shape
        new_data = np.zeros(new_shape)

        # TODO this can all probably be done smartly with np.reshape instead of a double for loop
        ind = 0
        inds = []
        hybs = []
        chs = []

        for h in range(self.image.num_hybs):
            for c in range(self.image.num_chs):
                new_data[ind, :] = self.image.numpy_array[h, c, :]
                inds.append(ind)
                hybs.append(h)
                chs.append(c)
                ind += 1

        self.squeeze_map = pd.DataFrame({'ind': inds, 'hyb': hybs, 'ch': chs})

        if bit_map_flag:
            mp = [(d['hyb'], d['ch'], d['bit']) for d in self.org['data']]
            mp = pd.DataFrame(mp, columns=['hyb', 'ch', 'bit'])
            self.squeeze_map = pd.merge(self.squeeze_map, mp, on=['ch', 'hyb'], how='left')

        return new_data

    def un_squeeze(self, stack):
        if type(stack) is list:
            stack = list_to_stack(stack)

        new_shape = (self.image.num_hybs, self.image.num_chs) + self.image.tile_shape
        res = np.zeros(new_shape)

        # TODO this can probably done smartly without a double for loop
        ind = 0
        for h in range(self.image.num_hybs):
            for c in range(self.image.num_chs):
                res[h, c, :] = stack[ind, :]
                ind += 1

        return res
