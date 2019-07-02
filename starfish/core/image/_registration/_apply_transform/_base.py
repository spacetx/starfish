from abc import abstractmethod
from typing import Type


from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline import PipelineComponent
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class ApplyTransform(PipelineComponent):
    """
    ApplyTransform exposes methods to align image data by transforming (and re-sampling if sub-pixel
    shifts are passed) input data according to a provided transformation.
    """
    pass


class ApplyTransformBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return ApplyTransform

    @abstractmethod
    def run(self, stack, transforms_list, *args) -> ImageStack:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
