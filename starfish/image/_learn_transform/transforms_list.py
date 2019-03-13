import json
from typing import List, Mapping, Tuple

import numpy as np
from skimage.transform._geometric import SimilarityTransform
from slicedimage.io import resolve_path_or_url

from starfish.config import StarfishConfig
from starfish.types import Axes


class TransformsList:
    """Simple list wrapper class for storing a list of transformation
    objects to apply to an Imagestack"""

    def __init__(self,
                 transforms_list: List[Tuple[Mapping[Axes, int], SimilarityTransform]] = None
                 ):
        """
        Parameters
        ----------
        transforms_list:  A list of tuples containing the axis of the Imagestack and
        associated transform to apply.
        """

        if transforms_list:
            self.transforms = transforms_list
        else:
            self.transforms: List[Tuple[Mapping[Axes, int], SimilarityTransform]] = list()

    def append(self,
               selectors: Mapping[Axes, int], transform_object: SimilarityTransform
               ) -> None:
        self.transforms.append((selectors, transform_object))

    def to_json(self, filename: str) -> None:
        """
        Saves the TransformsList to a json file.

        Parameters
        ----------
        filename : str
            filename
        """
        tranforms_array = []
        # Need to convert Axes to str values, and TransformationObjects to array
        for selectors, transforms_object in self.transforms:
            transforms_matrix = transforms_object.params.tolist()
            selectors_str = {k.value: v for k, v in selectors.items()}
            tranforms_array.append((selectors_str, transforms_matrix))
        with open(filename, 'w') as f:
            json.dump(tranforms_array, f)
        return

    @classmethod
    def from_json(cls, filename: str) -> "TransformsList":
        """
        Load the TransformsList from a json file or a url pointing to such a file
        Loads configuration from StarfishConfig.

        Parameters
        ----------
        filename : str
            filename

        Returns
        -------
        TransformsList
        """
        config = StarfishConfig()
        transforms_list: List[Tuple[Mapping[Axes, int], SimilarityTransform]] = list()
        backend, name, _ = resolve_path_or_url(filename, backend_config=config.slicedimage)
        with backend.read_contextmanager(name) as fh:
            transforms_array = json.load(fh)
        for selectors_str, transforms_matrix in transforms_array:
            selectors = {Axes(k): int(v) for k, v in selectors_str.items()}
            transform_object = SimilarityTransform(np.array(transforms_matrix))
            transforms_list.append((selectors, transform_object))
        return cls(transforms_list)
