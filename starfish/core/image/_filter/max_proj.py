from typing import Iterable, Optional, Union

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes
from starfish.core.util import click
from ._base import FilterAlgorithmBase


class MaxProject(FilterAlgorithmBase):
    """
    Creates a maximum projection over one or more axis of the image tensor

    Parameters
    ----------
    dims : Axes
        one or more Axes to project over

    See Also
    --------
    starfish.types.Axes

    """

    def __init__(self, dims: Iterable[Union[Axes, str]]) -> None:

        self.dims = dims

    _DEFAULT_TESTING_PARAMETERS = {"dims": 'r'}

    def run(
            self,
            stack: ImageStack,
            in_place: bool = False,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args,
    ) -> ImageStack:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on filtering progress (default = False)
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack. Otherwise return the
            original stack.

        """
        return stack.max_proj(*tuple(Axes(dim) for dim in self.dims))

    @staticmethod
    @click.command("MaxProject")
    @click.option(
        "--dims",
        type=click.Choice(
            [Axes.ROUND.value, Axes.CH.value, Axes.ZPLANE.value, Axes.X.value, Axes.Y.value]
        ),
        multiple=True,
        help="The dimensions the Imagestack should max project over."
             "For multiple dimensions add multiple --dims. Ex."
             "--dims r --dims c")
    @click.pass_context
    def _cli(ctx, dims):
        ctx.obj["component"]._cli_run(ctx, MaxProject(dims))
