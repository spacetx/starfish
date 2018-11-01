from typing import Type

import click

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, PipelineComponent
from . import fourier_shift
from ._base import RegistrationAlgorithmBase


class Registration(PipelineComponent):

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return RegistrationAlgorithmBase

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        stack = ctx.obj["stack"]
        instance.run(stack)
        stack.write(output)

    @staticmethod
    @click.group("registration")
    @click.option("-i", "--input", type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, input, output):
        print("Registering...")
        ctx.obj = dict(
            component=Registration,
            input=input,
            output=output,
            stack=ImageStack.from_path_or_url(input),
        )


Registration._cli_register()
