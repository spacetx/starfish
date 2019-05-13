from copy import deepcopy
from typing import Iterable, Optional, Union

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes
from starfish.core.util import click
from ._base import FilterAlgorithmBase


class Reduce(FilterAlgorithmBase):
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

    def __init__(self, dims: Iterable[Union[Axes, str]], func:str='max') -> None:

        self.dims = dims

        # If the user provided a string, convert to callable
        if isinstance(func, str):
            if func == 'max':
                func = np.amax
            elif func == 'mean':
                func = np.mean
            elif func == 'sum':
                func = np.sum
            else:
                raise ValueError('func should be max, mean, or sum')
        self.func = func

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
        if not in_place:
            stack = deepcopy(stack)

        # Apply the reducing function
        reduced = stack._data.reduce(self.func, dim=[dim.value for dim in self.dims])

        # Add the reduced dims back and align with the original stack
        reduced = reduced.expand_dims(tuple(dim.value for dim in self.dims))
        reduced = reduced.transpose(*stack.xarray.dims)

        # Construct the stack
        stack = stack.from_numpy(reduced.values)

        return stack


    @staticmethod
    @click.command("Reduce")
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
        ctx.obj["component"]._cli_run(ctx, Reduce(dims))
