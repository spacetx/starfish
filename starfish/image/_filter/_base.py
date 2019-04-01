from abc import abstractmethod
from typing import Type

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util import click


COMPONENT_NAME = "filter"


class Filter(PipelineComponent):
    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        stack = ctx.obj["stack"]
        filtered = instance.run(stack)
        filtered.export(output)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("-i", "--input", type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, input, output):
        """smooth, sharpen, denoise, etc"""
        print("Filtering images...")
        ctx.obj = dict(
            component=Filter,
            input=input,
            output=output,
            stack=ImageStack.from_path_or_url(input),
        )


class FilterAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return Filter

    @abstractmethod
    def run(self, stack: ImageStack, *args) -> ImageStack:
        """Performs filtering on an ImageStack."""
        raise NotImplementedError()
