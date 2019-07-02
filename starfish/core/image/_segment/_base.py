from abc import abstractmethod
from typing import Type

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline import PipelineComponent
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.segmentation_mask import SegmentationMaskCollection


class Segment(PipelineComponent):
    """
    Starfish class implementing segmentation approaches.
    """
    pass


class SegmentAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return Segment

    @abstractmethod
    def run(
            self,
            primary_image_stack: ImageStack,
            nuclei_stack: ImageStack,
            *args
    ) -> SegmentationMaskCollection:
        """Performs segmentation on the stack provided."""
        raise NotImplementedError()
