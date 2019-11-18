from abc import abstractmethod

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class SegmentAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(
            self,
            primary_image_stack: ImageStack,
            nuclei_stack: ImageStack,
            *args
    ) -> BinaryMaskCollection:
        """Performs segmentation on the stack provided."""
        raise NotImplementedError()
