from abc import abstractmethod

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class FilterAlgorithmBase(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, stack: ImageStack, *args) -> ImageStack:
        """Perform filtering of an image stack"""
        raise NotImplementedError()
