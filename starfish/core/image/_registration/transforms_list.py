import json
from typing import List, Mapping, Tuple

import numpy as np
from semantic_version import Version
from skimage.transform._geometric import GeometricTransform, SimilarityTransform
from slicedimage.io import resolve_path_or_url

from starfish.core.config import StarfishConfig
from starfish.core.image._registration._format import (
    CURRENT_VERSION,
    DocumentKeys,
    MAX_SUPPORTED_VERSION,
    MIN_SUPPORTED_VERSION,
)
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

        self.transforms: List[Tuple[Mapping[Axes, int], TransformType, GeometricTransform]]
        if transforms_list:
            self.transforms = transforms_list
        else:
            self.transforms = list()

    def __repr__(self) -> str:
        translation_strings = [
            f"tile indices: {t[0]}\ntranslation: y={t[2].translation[0]}, "
            f"x={t[2].translation[1]}, rotation: {t[2].rotation}, scale: {t[2].scale}"
            for t in self.transforms
        ]
        return "\n".join(translation_strings)

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

    @classmethod
    def _verify_version(cls, semantic_version_str: str) -> None:
        version = Version(semantic_version_str)
        if not (MIN_SUPPORTED_VERSION <= version <= MAX_SUPPORTED_VERSION):
            raise ValueError(
                f"version {version} not supported.  This version of the starfish library only "
                f"supports transform list formats from {MIN_SUPPORTED_VERSION} to "
                f"{MAX_SUPPORTED_VERSION}")

    def to_dict(self) -> dict:
        """
        Save the TransformsList as a Python dictionary.

        Returns
        -------
        dict
            A Python dictionary with the following keys and values:
                {DocumentKeys.TRANSFORMS_LIST}: List[Tuple[str, str, List[List[float]]]]
                    The value is a tuple of:
                        selector_string: A string that specifies which starfish dimensions
                                         to apply the transform to (e.g. "{r: 0}")
                        transform_type: Specifies the transform type (e.g. "similarity")
                        transform_matrix: A list of lists of floats
                {DocumentKeys.VERSION_KEY}: str
        """
        transforms_array = []
        # Need to convert Axes to str values, and TransformationObjects to array
        for selectors, transform_type, transforms_object in self.transforms:
            transforms_matrix = transforms_object.params.tolist()
            selectors_str = {k.value: v for k, v in selectors.items()}
            transforms_array.append((selectors_str, transform_type.value, transforms_matrix))
        transforms_document = {
            DocumentKeys.TRANSFORMS_LIST: transforms_array,
            DocumentKeys.VERSION_KEY: str(CURRENT_VERSION)
        }
        return transforms_document

    def to_json(self, filename: str) -> None:
        """
        Saves the TransformsList to a json file.

        Parameters
        ----------
        filename : str
        """
        transforms_document = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(transforms_document, f)

    @classmethod
    def from_dict(cls, transforms_document: dict) -> "TransformsList":
        """
        Load a TransformsList from a Python dictionary.

        Returns
        -------
        TransformsList
        """
        version_str = transforms_document[DocumentKeys.VERSION_KEY]
        cls._verify_version(version_str)
        transforms_array = transforms_document[DocumentKeys.TRANSFORMS_LIST]

        transforms_list: List[Tuple[Mapping[Axes, int], TransformType, GeometricTransform]] = list()
        for selectors_str, transform_type_str, transforms_matrix in transforms_array:
            selectors = {Axes(k): v for k, v in selectors_str.items()}
            transform_type = TransformType(transform_type_str)
            transform_object = transformsTypeMapping[transform_type](np.array(transforms_matrix))
            transforms_list.append((selectors, transform_type, transform_object))
        return cls(transforms_list)

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
        backend, name, _ = resolve_path_or_url(url_or_path, backend_config=config.slicedimage)
        with backend.read_contextmanager(name) as fh:
            transforms_document = json.load(fh)
        return cls.from_dict(transforms_document=transforms_document)
