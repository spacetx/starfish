import pandas as pd
from skimage import io
import numpy as np


class Stack:
    def __init__(self, fov_path, aux_path):
        self.fov_path = fov_path
        self.aux_path = aux_path

        self.fov_df = None
        self.aux_df = None

        self.stack = None
        self.num_hybs = None
        self.num_chans = None
        self.dims = None
        self.im_shape = None
        self.is_volume = None

        self.dapi = None
        self.aux_dict = dict()

    def read(self):
        self._read_fov()
        self._read_aux()

    def _read_fov(self):
        self.fov_df = pd.read_csv(self.fov_path)
        self.num_hybs = self.fov_df.hyb.max()
        self.num_chans = self.fov_df.ch.max()

        # correct for off by one error
        if self.fov_df.hyb.min() == 0:
            self.num_hybs += 1
        elif self.fov_df.hyb.min() == 1:
            self.fov_df.hyb -= 1

        if self.fov_df.ch.min() == 0:
            self.num_chans += 1
        elif self.fov_df.ch.min() == 1:
            self.fov_df.ch -= 1

        # determine image shape, set volumetric flag
        im = io.imread(self.fov_df.file[0])
        self.im_shape = im.shape

        if len(self.im_shape) == 2:
            self.stack = np.zeros((self.num_hybs, self.num_chans, self.im_shape[0], self.im_shape[1]))
            self.is_volume = False
        else:
            self.stack = np.zeros((self.num_hybs, self.num_chans, self.im_shape[0], self.im_shape[1], self.im_shape[2]))
            self.is_volume = True

        org = zip(self.fov_df.hyb.values, self.fov_df.ch.values, self.fov_df.file)

        for h, c, fname in org:
            self.stack[h, c, :] = io.imread(fname)

    def _read_aux(self):
        self.aux_df = pd.read_csv(self.aux_path)
        dapi_path = self.aux_df[self.aux_df.type == 'dapi'].file.values[0]
        self.dapi = io.imread(dapi_path)

        org = zip(self.aux_df.type.values, self.aux_df.file.values)

        for typ, fname in org:
            if typ == 'dapi':
                self.dapi = io.imread(fname)
            else:
                self.aux_dict[typ] = io.imread(fname)

    def write(self, fov_path, aux_path):
        self._write_fov(fov_path)
        self._write_aux(aux_path)

    def _write_fov(self, fov_path):
        pass

    def _write_aux(self, aux_path):
        pass

    def max_proj(self, dim):
        valid_dims = ['hyb', 'ch', 'z']
        if dim not in valid_dims:
            msg = "Dimension: {} not supported. Expecting one of: {}".format(dim, valid_dims)
            raise ValueError(msg)

        if dim == 'hyb':
            res = np.max(self.stack, axis=0)
        elif dim == 'ch':
            res = np.max(self.stack, axis=1)
        elif dim == 'z' and self.is_volume:
            res = np.max(self.stack, axis=4)
        else:
            res = self.stack

        return res

    def combine_hyb_ch(self):
        pass
