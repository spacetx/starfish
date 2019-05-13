from copy import deepcopy
from typing import Callable, Iterable, Optional, Union

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes
from starfish.core.util import click
from ._base import FilterAlgorithmBase


class Reduce(FilterAlgorithmBase):
    """
    Reduces the dimensions of the ImageStack by applying a function
    along one or more axes.

    Parameters
    ----------
    dims : Axes
        one or more Axes to project over
    func : Union[str, Callable]
        function to apply across the dimension(s) specified by dims.
        If a function is provided, it should follow the form specified by
        DataArray.reduce():
        http://xarray.pydata.org/en/stable/generated/xarray.DataArray.reduce.html

        The following strings are valid:
            max: maximum intensity projection (applies numpy.amax)
            mean: take the mean across the dim(s) (applies numpy.mean)
            sum: sum across the dim(s) (applies numpy.sum)

        Note: user-specified functions are not recorded via starfish logging

    See Also
    --------
    starfish.types.Axes

    """

    def __init__(
        self, dims: Iterable[Union[Axes, str]], func: Union[str, Callable] = 'max'
    ) -> None:

        self.dims = dims

        # If the user provided a string, convert to callable
        if isinstance(func, str):
            if func == 'max':
                func = 'amax'
            func = getattr(np, func)
        self.func = func

    _DEFAULT_TESTING_PARAMETERS = {"dims": ['r'], "func": 'max'}

    def run(
            self,
            stack: ImageStack,
            in_place: bool = False,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args,
    ) -> ImageStack:
        """Performs the dimension reduction with the specifed function

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
    @click.option(
        "--func",
        type=click.Choice(["max", "mean", "sum"]),
        multiple=False,
        help="The function to apply across dims"
             "Valid function names: max, mean, sum."
    )
    @click.pass_context
    def _cli(ctx, dims, func):
        ctx.obj["component"]._cli_run(ctx, Reduce(dims, func))
