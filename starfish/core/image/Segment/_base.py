from abc import abstractmethod

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.segmentation_mask import SegmentationMaskCollection


class SegmentAlgorithmBase(metaclass=AlgorithmBase):

    @abstractmethod
    def run(
            self,
            primary_image_stack: ImageStack,
            nuclei_stack: ImageStack,
            *args
    ) -> SegmentationMaskCollection:
        """Performs segmentation on the stack provided."""
        raise NotImplementedError()
