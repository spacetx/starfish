import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from slicedimage import ImageFormat
from slicedimage.io import resolve_url, resolve_path_or_url

from starfish.constants import Indices
from .image import ImageStack
from .munge import list_to_stack


class Stack:
    def __init__(self):
        # data organization
        self.org = None
        self.image = None

        # auxilary images
        self.aux_dict = dict()

        # readers and writers
        self.write_fn = np.save  # asserted for now

        # backend & baseurl
        self.backend = None
        self.baseurl = None

    def read(self, in_json_path_or_url):
        self.backend, name, self.baseurl = resolve_path_or_url(in_json_path_or_url)
        with self.backend.read_file_handle(name) as fh:
            self.org = json.load(fh)

        self.image = ImageStack.from_url(self.org['hybridization_images'], self.baseurl)
        self._read_aux()

    def _read_aux(self):
        for aux_key, aux_data in self.org['auxiliary_images'].items():
            name_or_url = aux_data['file']
            img_format = ImageFormat[aux_data['tile_format']]
            backend, name, _ = resolve_url(name_or_url, self.baseurl)
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
        if self.image.raw_shape != new_stack.shape:
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

    def max_proj(self, *dims):
        return self.image.max_proj(*dims)

    def squeeze(self, bit_map_flag=False):
        first_dim = self.image.num_hybs * self.image.num_chs * self.image.num_zlayers

        new_shape = (first_dim,) + self.image.tile_shape
        new_data = self.image.numpy_array.reshape(new_shape)

        ind = np.arange(first_dim)

        # e.g., 0, 1, 2, 3 --> 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
        hyb = np.tile(np.repeat(np.arange(self.image.num_hybs), self.image.num_chs), self.image.num_zlayers)

        # e.g., 0, 1, 2, 3 --> 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
        ch = np.tile(np.arange(self.image.num_chs), self.image.num_hybs * self.image.num_zlayers)

        # e.g., 0, 1, 2, 3 --> 0 (repeated 16 times), 1 (repeated 16 times), ...
        zlayers = np.repeat(np.arange(self.image.num_zlayers), self.image.num_hybs * self.image.num_chs)

        self.squeeze_map = pd.DataFrame(dict(ind=ind, hyb=hyb, ch=ch, zlayers=zlayers))

        if bit_map_flag:
            mp = [(d[Indices.HYB], d[Indices.CH], d['bit']) for d in self.org['data']]
            mp = pd.DataFrame(mp, columns=[Indices.HYB, Indices.CH, 'bit'])
            self.squeeze_map = pd.merge(self.squeeze_map, mp, on=[Indices.CH, Indices.HYB], how='left')

        return new_data

    def un_squeeze(self, stack):
        if type(stack) is list:
            stack = list_to_stack(stack)

        new_shape = (self.image.num_hybs, self.image.num_chs, self.image.num_zlayers) + self.image.tile_shape
        res = stack.reshape(new_shape)
        return res

    def show_z_stack(
            self, h: int=1, c: int=0, cmap: str='gray', rescale: bool=False, figsize: Tuple[int, int]=(10, 10)):
        """Create an interactive visualization of a z-stack

        Produces a slider that flips through the z-series of 2-d x-y images

        Parameters
        ----------
        h : int
            hybridization round to visualize
        c : int
            channel to visualize
        cmap : str (default = 'gray')
            string id of a matplotlib colormap
        rescale : bool (default = True)
            if True, rescale the data to exclude high and low-value outliers (see skimage.exposure.rescale_intensity)
        figsize : Tuple[int, int] (default = (10, 10))
            size of the figure in inches

        """
        from skimage import exposure

        from starfish.constants import Indices

        import matplotlib.pyplot as plt
        from ipywidgets import interact
        from scipy.stats import scoreatpercentile

        n = self.image.shape[Indices.Z]

        data = self.image.get_slice({Indices.CH: c, Indices.HYB: h})[0]
        if rescale:
            print("Rescaling ...")
            vmin, vmax = scoreatpercentile(data, (0.5, 99.5))
            data = exposure.rescale_intensity(
                data,
                in_range=(vmin, vmax),
                out_range=np.float32
            ).astype(np.float32)

        def show_plane(ax, plane, cmap="gray", title=None):
            ax.imshow(plane, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])

            if title:
                ax.set_title(title)

        @interact(plane=(0, n - 1))
        def display_slice(plane=34):
            fig, ax = plt.subplots(figsize=figsize)
            show_plane(ax, data[plane], title="Plane {}".format(plane), cmap=cmap)
            plt.show()

        return display_slice
