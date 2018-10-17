import click
from typing import Type

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, PipelineComponent
from starfish.util.argparse import FsExistsType
from . import fourier_shift
from ._base import RegistrationAlgorithmBase


class Registration(PipelineComponent):

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return RegistrationAlgorithmBase

    @classmethod
    @click.group(name="registration")
    @click.option("-i", "--input")  # FIXME
    @click.option("o", "--output", required=True)
    @click.pass_context
    def _cli(cls, ctx, input, output):
        print('Registering ...')
        ctx.obj = ImageStack.from_path_or_url(args.input)


for algorithm_cls in Registration._algorithm_to_class_map().values():
    Registration._cli.add_command(algorithm_cls._cli)
