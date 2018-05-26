import json
import os
from collections import defaultdict

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

    def squeeze(self) -> np.ndarray:
        """return an array that is linear over categorical dimensions and z

        Returns
        -------
        np.ndarray :
            array of shape (num_hybs + num_channels + num_z_layers, x, y).

        """
        first_dim = self.image.num_hybs * self.image.num_chs * self.image.num_zlayers
        new_shape = (first_dim,) + self.image.tile_shape
        new_data = self.image.numpy_array.reshape(new_shape)

        return new_data

    @property
    def tile_metadata(self) -> pd.DataFrame:
        """return a table containing Tile metadata

        Returns
        -------
        pd.DataFrame :
            dataframe containing per-tile metadata information for each image. Guaranteed to include information on
            channel, hybridization round, z_layer, and barcode index. Also contains any information stored in the
            extras field for each tile in hybridization.json

        """

        data = defaultdict(list)
        for tile in self.image._image_partition.tiles():
            for k, v in tile.indices.items():
                data[k].append(v)
            for k, v in tile.extras.items():
                data[k].append(v)

        if 'barcode_index' not in data:
            data['barcode_index'] = np.arange(self.image.num_hybs * self.image.num_chs * self.image.num_zlayers)

        return pd.DataFrame(data)

    def un_squeeze(self, stack):
        if type(stack) is list:
            stack = list_to_stack(stack)

        new_shape = (self.image.num_hybs, self.image.num_chs, self.image.num_zlayers) + self.image.tile_shape
        res = stack.reshape(new_shape)
        return res
