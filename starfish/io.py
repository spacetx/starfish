import json
import os
from typing import Optional, Mapping

import numpy as np
from slicedimage import TileSet, Tile
from slicedimage.io import resolve_path_or_url

from starfish.constants import Coordinates, Indices
from .image import ImageStack


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

    @classmethod
    def from_data(cls, image_stack: ImageStack, aux_dict: Optional[Mapping[str, ImageStack]]=None) -> "Stack":
        """create a Stack from an already-loaded ImageStack

        Parameters
        ----------
        image_stack : ImageStack
            in-memory ImageStack
        aux_dict : Optional[Mapping[str, ImageStack]]
            a dictionary of ImageStacks, default None

        Returns
        -------
        Stack :
            a Stack object

        """
        stack = cls()
        stack.image = image_stack
        stack.auxiliary_images = aux_dict if aux_dict is not None else dict()
        return stack

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
    def from_experiment_json(cls, json_url: str) -> "Stack":
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
        self.org['hybridization_images'] = "hybridization.json"

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
