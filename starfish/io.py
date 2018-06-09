import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from slicedimage import ImageFormat, TileSet, Tile
from slicedimage.io import resolve_path_or_url, Writer

from starfish.constants import Coordinates, Indices
from .image import ImageStack
from .munge import list_to_stack


class Stack:

    def __init__(self):
        # data organization
        self.org = None
        self.image = None

        # auxiliary images
        self.auxiliary_images = dict()

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
            self.auxiliary_images[aux_key] = ImageStack.from_url(aux_data, self.baseurl)

    @classmethod
    def from_experiment_json(cls, json_url: str):
        """Construct a `Stack` from an experiment.json file format specifier

        Parameters
        ----------
        json_url : str
            file path or web link to an experiment.json file

        Returns
        -------
        Stack :
            Stack object serving the requested image data

        """
        stack: Stack = cls()
        stack.read(json_url)
        return stack

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
            self.auxiliary_images[aux_key].write(os.path.join(dir_name, aux_data))

    def set_stack(self, new_stack):
        if self.image.raw_shape != new_stack.shape:
            msg = "Shape mismatch. Current data shape: {}, new data shape: {}".format(
                self.image.shape, new_stack.shape)
            raise AttributeError(msg)
        self.image.numpy_array = new_stack

    def set_aux(self, key, img):
        if key in self.auxiliary_images:
            old_img = self.auxiliary_images[key]
            if old_img.shape != img.shape:
                msg = "Shape mismatch. Current data shape: {}, new data shape: {}".format(
                    old_img.shape, img.shape)
                raise AttributeError(msg)
            self.auxiliary_images[key].numpy_array = img
        else:
            # TODO: (ttung) major hack alert.  we don't have a convenient mechanism to build an ImageStack from a single
            # numpy array, which we probably should.
            tileset = TileSet(
                {
                    Indices.HYB,
                    Indices.CH,
                    Indices.Z,
                    Coordinates.X,
                    Coordinates.Y,
                },
                {
                    Indices.HYB: 1,
                    Indices.CH: 1,
                    Indices.Z: 1,
                }
            )
            tile = Tile(
                {
                    Coordinates.X: (0.000, 0.001),
                    Coordinates.Y: (0.000, 0.001),
                },
                {
                    Indices.HYB: 0,
                    Indices.CH: 0,
                    Indices.Z: 0,
                },
                img.shape,
            )
            tile.numpy_array = img
            tileset.add_tile(tile)

            self.auxiliary_images[key] = ImageStack(tileset)
            self.org['auxiliary_images'][key] = f"{key}.json"

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
        data: defaultdict = defaultdict(list)
        index_keys = set(
            key
            for tile in self.image._image_partition.tiles()
            for key in tile.indices.keys())
        extras_keys = set(
            key
            for tile in self.image._image_partition.tiles()
            for key in tile.extras.keys())
        duplicate_keys = index_keys.intersection(extras_keys)
        if len(duplicate_keys) > 0:
            duplicate_keys_str = ", ".join(duplicate_keys)
            raise ValueError(
                f"keys ({duplicate_keys_str}) was found in both the Tile specification and extras field. Tile "
                f"specification keys may not be duplicated in the extras field.")

        for tile in self.image._image_partition.tiles():
            for k in index_keys:
                data[k].append(tile.indices.get(k, None))
            for k in extras_keys:
                data[k].append(tile.extras.get(k, None))

            if 'barcode_index' not in tile.extras:
                hyb = tile.indices[Indices.HYB]
                ch = tile.indices[Indices.CH]
                z = tile.indices.get(Indices.Z, 0)
                barcode_index = (((z * self.image.num_hybs) + hyb) * self.image.num_chs) + ch

                data['barcode_index'].append(barcode_index)

        return pd.DataFrame(data)

    def un_squeeze(self, stack):
        if type(stack) is list:
            stack = list_to_stack(stack)

        new_shape = (self.image.num_hybs, self.image.num_chs, self.image.num_zlayers) + self.image.tile_shape
        res = stack.reshape(new_shape)
        return res
