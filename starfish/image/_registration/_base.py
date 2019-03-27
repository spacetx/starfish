from abc import abstractmethod
from typing import Type

import click

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import PipelineComponent
from starfish.pipeline.algorithmbase import AlgorithmBase


COMPONENT_NAME = "registration"


class Registration(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        stack = ctx.obj["stack"]
        instance.run(stack)
        stack.export(output)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("-i", "--input", type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, input, output):
        """translation correction of image stacks"""
        print("Registering...")
        ctx.obj = dict(
            component=Registration,
            input=input,
            output=output,
            stack=ImageStack.from_path_or_url(input),
        )


class RegistrationAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return Registration

    @abstractmethod
    def run(self, stack: ImageStack, *args) -> ImageStack:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
