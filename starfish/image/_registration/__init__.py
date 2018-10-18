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
    @click.group("registration")
    @click.option("-i", "--input")  # FIXME
    @click.option("o", "--output", required=True)
    @click.pass_context
    def _cli(cls, ctx, input, output):
        print("Registering...")
        ctx.stack = ImageStack.from_path_or_url(input)

    @classmethod
    def _cli_run(cls, ctx, instance):
        instance.run(ctx.stack)
        ctx.stack.write(ctx.ouput)


Registration._cli_register()
