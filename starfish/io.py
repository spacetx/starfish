import pandas as pd
from skimage import io
import numpy as np

from .munge import list_to_stack
import json
import os


class Stack:
    def __init__(self):

        # data organization
        self.org = None
        self.path = None
        self.format = None

        # numpy array (num_hybs, num_chans, x, y, z)
        self.data = None

        # auxilary images
        self.aux_dict = dict()

        # shape data
        self.num_hybs = None
        self.num_chs = None
        self.im_shape = None
        self.is_volume = None
        self.squeeze_map = None

        # readers and writers
        self.read_fn = None  # set by self._read_metadata
        self.write_fn = np.save  # asserted for now

    @property
    def shape(self):
        if self.data is None:
            return None
        else:
            return self.data.shape

    def read(self, in_json):
        with open(in_json, 'r') as in_file:
            self.org = json.load(in_file)

        self.path = os.path.dirname(os.path.abspath(in_json)) + '/'
        self._read_metadata()
        self._read_stack()
        self._read_aux()

    def _read_metadata(self):
        d = self.org['metadata']
        self.num_hybs = d['num_hybs']
        self.num_chs = d['num_chs']
        self.im_shape = tuple(d['shape'])

        self.is_volume = d['is_volume']
        if not self.is_volume:
            self.data = np.zeros((self.num_hybs, self.num_chs, self.im_shape[0], self.im_shape[1]))
        else:
            self.data = np.zeros((self.num_hybs, self.num_chs, self.im_shape[0], self.im_shape[1], self.im_shape[2]))

        self.format = d['format']

        if self.format == 'tiff':
            self.read_fn = io.imread
        else:
            self.read_fn = np.load

    def _read_stack(self):
        data_dicts = self.org['data']

        for d in data_dicts:
            h = d['hyb']
            c = d['ch']
            fname = d['file']
            im = self.read_fn(os.path.join(self.path, fname))
            self.data[h, c, :] = im

    def _read_aux(self):
        data_dicts = self.org['aux']

        for d in data_dicts:
            typ = d['type']
            fname = d['file']
            self.aux_dict[typ] = self.read_fn(os.path.join(self.path, fname))

    # TODO should this thing write npy?
    def write(self, dir_name):
        self._write_metadata(dir_name)
        self._write_stack(dir_name)
        self._write_aux(dir_name)

    def _write_metadata(self, dir_name):
        new_org = self.org
        new_org['metadata']['format'] = 'npy'

        def format(d):
            d['file'] = self._swap_ext(d['file']) + '.npy'
            return d

        new_org['data'] = [format(d) for d in new_org['data']]
        new_org['aux'] = [format(d) for d in new_org['aux']]

        with open(os.path.join(dir_name, 'org.json'), 'w') as outfile:
            json.dump(new_org, outfile, indent=4)

        self.org = new_org

    def _write_stack(self, dir_name):
        for d in self.org['data']:
            h = d['hyb']
            c = d['ch']
            fname = d['file']
            self.write_fn(os.path.join(dir_name, fname), self.data[h, c, :])

    def _write_aux(self, dir_name):
        for d in self.org['aux']:
            typ = d['type']
            fname = d['file']
            self.write_fn(os.path.join(dir_name, fname), self.aux_dict[typ])

    def _swap_ext(self, fname):
        if self.format == 'tiff':
            fname = fname.replace('.tiff', '')
        else:
            fname = fname.replace('.npy', '')
        return fname

    def set_stack(self, new_stack):
        if new_stack.shape != self.shape:
            msg = "Shape mismatch. Current data shape: {}, new data shape: {}".format(self.shape, new_stack.shape)
            raise AttributeError(msg)
        self.data = new_stack

    def set_aux(self, key, img):
        if key in self.aux_dict:
            old_stack = self.aux_dict[key]
            if old_stack.shape != img.shape:
                msg = "Shape mismatch. Current data shape: {}, new data shape: {}".format(old_stack.shape,
                                                                                          img.shape)
                raise AttributeError(msg)
        else:
            self.org['aux'].append({'file': key, 'type': key})
        self.aux_dict[key] = img

    def max_proj(self, dim):
        valid_dims = ['hyb', 'ch', 'z']
        if dim not in valid_dims:
            msg = "Dimension: {} not supported. Expecting one of: {}".format(dim, valid_dims)
            raise ValueError(msg)

        if dim == 'hyb':
            res = np.max(self.data, axis=0)
        elif dim == 'ch':
            res = np.max(self.data, axis=1)
        elif dim == 'z' and self.is_volume:
            res = np.max(self.data, axis=4)
        else:
            res = self.data

        return res

    def squeeze(self, bit_map_flag=False):
        new_shape = ((self.num_hybs * self.num_chs),) + self.im_shape
        new_data = np.zeros(new_shape)

        # TODO this can all probably be done smartly with np.reshape instead of a double for loop
        ind = 0
        inds = []
        hybs = []
        chs = []

        for h in range(self.num_hybs):
            for c in range(self.num_chs):
                new_data[ind, :] = self.data[h, c, :]
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

        new_shape = (self.num_hybs, self.num_chs) + self.im_shape
        res = np.zeros(new_shape)

        # TODO this can probably done smartly without a double for loop
        ind = 0
        for h in range(self.num_hybs):
            for c in range(self.num_chs):
                res[h, c, :] = stack[ind, :]
                ind += 1

        return res
