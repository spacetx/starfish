import json
from typing import Optional, Mapping

import numpy as np
from semantic_version import Version
from slicedimage.io import resolve_path_or_url

from .image import ImageStack


class Stack:
    MIN_SUPPORTED_VERSION = Version("1.0.0")
    MAX_SUPPORTED_VERSION = Version("1.0.0")

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

        self.verify_version(self.org['version'])
        self.image = ImageStack.from_url(self.org['hybridization_images'], self.baseurl)
        for aux_key, aux_data in self.org['auxiliary_images'].items():
            self.auxiliary_images[aux_key] = ImageStack.from_url(aux_data, self.baseurl)

    @classmethod
    def verify_version(cls, semantic_version_str: str):
        version = Version(semantic_version_str)
        if not (Stack.MIN_SUPPORTED_VERSION <= version <= Stack.MAX_SUPPORTED_VERSION):
            raise ValueError(
                f"version {version} not supported.  This version of the starfish library only "
                f"supports formats from {Stack.MIN_SUPPORTED_VERSION} to "
                f"{Stack.MAX_SUPPORTED_VERSION}")

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
