import json
from typing import List, Mapping, Tuple

import numpy as np
from skimage.transform._geometric import GeometricTransform, SimilarityTransform
from slicedimage.io import resolve_path_or_url

from starfish.core.config import StarfishConfig
from starfish.core.types import Axes, TransformType


transformsTypeMapping = {
    TransformType.SIMILARITY: SimilarityTransform
}


class TransformsList:
    """Simple list wrapper class for storing a list of transformation
    objects to apply to an Imagestack"""

    def __init__(self,
                 transforms_list: List[Tuple[Mapping[Axes, int],
                                             TransformType,
                                             GeometricTransform]] = None
                 ):
        """
        Parameters
        ----------
        transforms_list: List[Tuple[Mapping[Axes, int], TransformType, GeometricTransform]]
            A list of tuples containing axes of an Imagestack and associated
            transform to apply.
        """

        if transforms_list:
            self.transforms = transforms_list
        else:
            self.transforms: List[Tuple[Mapping[Axes, int],
                                        TransformType, GeometricTransform]] = list()

    def append(self,
               selectors: Mapping[Axes, int],
               transform_type: TransformType,
               transform_object: GeometricTransform
               ) -> None:
        """
        Adds a new GoemetricTransform object to the list

        Parameters
        ----------
        transform_object:
            The TransformationObject to add
        transform_type:
            The type of transform
        selectors:
            The axes associated with the transform.

        """
        self.transforms.append((selectors, transform_type, transform_object))

    def to_json(self, filename: str) -> None:
        """
        Saves the TransformsList to a json file.

        Parameters
        ----------
        filename : str
        """
        transforms_array = []
        # Need to convert Axes to str values, and TransformationObjects to array
        for selectors, transform_type, transforms_object in self.transforms:
            transforms_matrix = transforms_object.params.tolist()
            selectors_str = {k.value: v for k, v in selectors.items()}
            transforms_array.append((selectors_str, transform_type.value, transforms_matrix))
        with open(filename, 'w') as f:
            json.dump(transforms_array, f)

    @classmethod
    def from_json(cls, url_or_path: str) -> "TransformsList":
        """
        Load a TransformsList from a json file or a url pointing to such a file
        Loads slicedimage version configuration from :py:class:`starfish.config.StarfishConfig`

        Parameters
        ----------
        url_or_path : str
            Either an absolute URL or a filesystem path to a transformsList.

        Returns
        -------
        TransformsList
        """
        config = StarfishConfig()
        transforms_list: List[Tuple[Mapping[Axes, int], TransformType, GeometricTransform]] = list()
        backend, name, _ = resolve_path_or_url(url_or_path, backend_config=config.slicedimage)
        with backend.read_contextmanager(name) as fh:
            transforms_array = json.load(fh)
        for selectors_str, transform_type_str, transforms_matrix in transforms_array:
            selectors = {Axes(k): v for k, v in selectors_str.items()}
            transform_type = TransformType(transform_type_str)
            transform_object = transformsTypeMapping[transform_type](np.array(transforms_matrix))
            transforms_list.append((selectors, transform_type, transform_object))
        return cls(transforms_list)
