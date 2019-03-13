from typing import List, Mapping, Tuple

from skimage.transform._geometric import GeometricTransform

from starfish.types import Axes


class TransformsList:
    """Simple list wrapper class for storing a list of transformation
    objects to apply to an Imagestack"""

    def __init__(self):
        self.transforms: List[Tuple[Mapping[Axes, int], GeometricTransform]] = list()

    def append(self,
               selectors: Mapping[Axes, int], transform_object: GeometricTransform
               ) -> None:
        self.transforms.append((selectors, transform_object))

    def export(self, filepath: str) -> None:
        return

    @staticmethod
    def from_path(filepath: str) -> "TransformsList":
        return TransformsList()
