from abc import abstractmethod
from typing import Optional

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class FilterAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, stack: ImageStack, *args) -> Optional[ImageStack]:
        """Perform filtering of an image stack"""
        raise NotImplementedError()
