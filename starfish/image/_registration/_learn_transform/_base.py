from abc import abstractmethod
from typing import Type


from starfish.image._registration.transforms_list import TransformsList
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util import click
from starfish.util.click.indirectparams import ImageStackParamType

COMPONENT_NAME = "learn_transform"


class LearnTransform(PipelineComponent):
    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance, *args, **kwargs):
        output = ctx.obj["output"]
        stack = ctx.obj["stack"]
        transformation_list = instance.run(stack)
        transformation_list.to_json(output)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("-i", "--input", type=ImageStackParamType)
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, input, output):
        """Learn a set of transforms for an ImageStack."""
        print("Learning Transforms for images...")
        ctx.obj = dict(
            component=LearnTransform,
            output=output,
            stack=input,
        )


class LearnTransformBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return LearnTransform

    @abstractmethod
    def run(self, stack, *args) -> TransformsList:
        """Learns Transforms for a given stack."""
        raise NotImplementedError()
