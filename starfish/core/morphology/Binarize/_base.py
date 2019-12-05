from abc import abstractmethod

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class BinarizeAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, image: ImageStack, *args, **kwargs) -> BinaryMaskCollection:
        """Performs binarization on the stack provided."""
        raise NotImplementedError()
