from abc import abstractmethod
from typing import Type

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import PipelineComponent
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.util import click


COMPONENT_NAME = "apply_transform"

class ApplyTransform(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        stack = ctx.obj["stack"]
        transformed = instance.run(stack)
        transformed.export(output)

    @staticmethod
    @click.group("apply_transform")
    @click.option("-i", "--input", type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, input, output):
        print("Applying Transform to images...")
        ctx.obj = dict(
            component=ApplyTransform,
            input=input,
            output=output,
            stack=ImageStack.from_path_or_url(input),
        )


class ApplyTransformBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return ApplyTransform

    @abstractmethod
    def run(self, stack) -> ImageStack:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
