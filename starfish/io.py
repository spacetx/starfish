import pandas as pd
from skimage import io
import numpy as np

from starfish.munge import list_to_stack


class Stack:
    def __init__(self):

        # data organization, dataframes
        self.org = None
        self.aux_org = None

        # numpy array (num_hybs, num_chans, x, y, z)
        self.data = None

        # shape of array
        self.num_hybs = None
        self.num_chans = None
        self.im_shape = None
        self.is_volume = None

        # auxilary images
        self.dapi = None
        self.aux_dict = dict()

    def read(self, fov_path, aux_path):
        self._read_fov(fov_path)
        if aux_path is not None:
            self._read_aux(aux_path)

    def write(self, dir_name, ext):

        hybs = self.org.hyb.values
        chs = self.org.ch.values
        inds = zip(hybs, chs)

        fnames = []
        for h, c in inds:
            im = self.data[h, c, :]
            fname = dir_name + '/h{}_c{}.{}'.format(h, c, ext)
            io.imsave(fname, im)
            fnames.append(fname)

        org = pd.DataFrame({'file': fnames, 'hyb': hybs, 'ch': chs})
        org.to_csv(dir_name + '/org.csv', index=False)

    def _read_fov(self, fov_path):
        self.org = pd.read_csv(fov_path)
        self.num_hybs = self.org.hyb.max()
        self.num_chans = self.org.ch.max()

        # correct for off by one error
        if self.org.hyb.min() == 0:
            self.num_hybs += 1
        elif self.org.hyb.min() == 1:
            self.org.hyb -= 1

        if self.org.ch.min() == 0:
            self.num_chans += 1
        elif self.org.ch.min() == 1:
            self.org.ch -= 1

        # determine image shape, set volumetric flag
        im = io.imread(self.org.file[0])
        self.im_shape = im.shape

        if len(self.im_shape) == 2:
            self.data = np.zeros((self.num_hybs, self.num_chans, self.im_shape[0], self.im_shape[1]))
            self.is_volume = False
        else:
            self.data = np.zeros((self.num_hybs, self.num_chans, self.im_shape[0], self.im_shape[1], self.im_shape[2]))
            self.is_volume = True

        org = zip(self.org.hyb.values, self.org.ch.values, self.org.file)

        for h, c, fname in org:
            self.data[h, c, :] = io.imread(fname)

    def _read_aux(self, aux_path):
        self.aux_org = pd.read_csv(aux_path)
        dapi_path = self.aux_org[self.aux_org.type == 'dapi'].file.values[0]
        self.dapi = io.imread(dapi_path)

        org = zip(self.aux_org.type.values, self.aux_org.file.values)

        for typ, fname in org:
            if typ == 'dapi':
                self.dapi = io.imread(fname)
            else:
                self.aux_dict[typ] = io.imread(fname)

    @property
    def shape(self):
        return self.data.shape

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

    def squeeze(self):
        new_shape = ((self.num_hybs * self.num_chans),) + self.im_shape
        new_data = np.zeros(new_shape)

        # TODO this can probably done smartly with np.reshape instead of a double for loop
        ind = 0
        for h in range(self.num_hybs):
            for c in range(self.num_chans):
                new_data[ind, :] = self.data[h, c, :]
                ind += 1

        return new_data

    def un_squeeze(self, stack):
        if type(stack) is list:
            stack = list_to_stack(stack)

        new_shape = (self.num_hybs, self.num_chans) + self.im_shape
        res = np.zeros(new_shape)

        # TODO this can probably done smartly without a double for loop
        ind = 0
        for h in range(self.num_hybs):
            for c in range(self.num_chans):
                res[h, c, :] = stack[ind, :]
                ind += 1

        return res