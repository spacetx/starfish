from abc import abstractmethod
from typing import Type


from starfish.core.image._registration.transforms_list import TransformsList
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.pipeline.pipelinecomponent import PipelineComponent


class LearnTransform(PipelineComponent):
    """
    LearnTransform exposes methods to learn transformations that align ImageStacks.
    """
    pass


class LearnTransformBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return LearnTransform

    @abstractmethod
    def run(self, stack, *args) -> TransformsList:
        """Learns Transforms for a given stack."""
        raise NotImplementedError()
