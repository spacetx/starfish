from abc import abstractmethod
from typing import Type

from starfish.image._registration.transforms_list import TransformsList
from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import PipelineComponent
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.util import click
from starfish.util.click.indirectparams import ImageStackParamType

COMPONENT_NAME = "apply_transform"


class ApplyTransform(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        stack = ctx.obj["stack"]
        transformation_list = ctx.obj["transformation_list"]
        transformed = instance.run(stack, transformation_list)
        transformed.export(output)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("-i", "--input", type=ImageStackParamType)
    @click.option("-o", "--output", required=True)
    @click.option("--transformation-list", required=True, type=click.Path(exists=True),
                  help="The list of transformations to apply to the ImageStack.")
    @click.pass_context
    def _cli(ctx, input, output, transformation_list):
        print("Applying Transform to images...")
        ctx.obj = dict(
            component=ApplyTransform,
            output=output,
            stack=input,
            transformation_list=TransformsList.from_json(transformation_list)
        )


class ApplyTransformBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return ApplyTransform

    @abstractmethod
    def run(self, stack, transforms_list, *args) -> ImageStack:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
