from abc import abstractmethod

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class SegmentAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, *args, **kwargs) -> BinaryMaskCollection:
        """Performs segmentation."""
        raise NotImplementedError()
