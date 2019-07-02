from abc import abstractmethod
from typing import Type

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.pipeline.pipelinecomponent import PipelineComponent


class Filter(PipelineComponent):
    pass


class FilterAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return Filter

    @abstractmethod
    def run(self, stack: ImageStack, *args) -> ImageStack:
        """Perform filtering of an image stack"""
        raise NotImplementedError()
