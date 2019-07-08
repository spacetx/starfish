from abc import abstractmethod

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class ApplyTransformBase(AlgorithmBase):

    @abstractmethod
    def run(self, stack, transforms_list, *args) -> ImageStack:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
