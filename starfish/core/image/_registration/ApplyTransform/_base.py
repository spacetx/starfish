from abc import abstractmethod
from typing import Optional

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class ApplyTransformAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, stack, transforms_list, *args) -> Optional[ImageStack]:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
