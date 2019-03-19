from typing import Type

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, import_all_submodules, PipelineComponent
from starfish.util import click
from . import _base
import_all_submodules(__file__, __package__)


COMPONENT_NAME = "registration"


class Registration(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return _base.RegistrationAlgorithmBase

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
