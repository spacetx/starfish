from typing import Optional

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes
from starfish.util import click
from ._base import FilterAlgorithmBase


class MaxProj(FilterAlgorithmBase):

    def __init__(self, dims) -> None:
        self.dims = dims

    _DEFAULT_TESTING_PARAMETERS = {"dims": 'r'}

    def run(
            self, stack: ImageStack, in_place: bool = False, verbose: bool = False,
            n_processes: Optional[int] = None
    ) -> ImageStack:
        return stack.max_proj(*tuple(Axes(dim) for dim in self.dims))

    @staticmethod
    @click.command("MaxProj")
    @click.option("--dims", type=str, multiple=True,
                  help="The dimensions the Imagestack should max project over. Options:"
                       "(r, c, z, y, or x) For multiple dimensions add multiple --dims. Ex."
                       "--dims r --dims c")
    @click.pass_context
    def _cli(ctx, dims):
        ctx.obj["component"]._cli_run(ctx, MaxProj(dims))
