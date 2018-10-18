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
        stack = ctx.obj["stack"]
        instance.run(stack)
        stack.write(ctx.ouput)

@click.group("registration")
@click.option("-i", "--input")  # FIXME
@click.option("o", "--output", required=True)
@click.pass_context
def _cli(ctx, input, output):
    print("Registering...")
    ctx.obj = dict(
        component=Registration,
        stack=ImageStack.from_path_or_url(input),
    )


Registration._cli = _cli
Registration._cli_register()
