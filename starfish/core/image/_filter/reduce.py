from copy import deepcopy
from typing import Callable, Iterable, Optional, Union

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, Clip
from starfish.core.util import click
from starfish.core.util.dtype import preserve_float_range
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

    clip_method : Union[str, Clip]
        (Default Clip.CLIP) Controls the way that data are scaled to retain skimage dtype
        requirements that float data fall in [0, 1].
        Clip.CLIP: data above 1 are set to 1, and below 0 are set to 0
        Clip.SCALE_BY_IMAGE: data above 1 are scaled by the maximum value, with the maximum
        value calculated over the entire ImageStack
        Clip.SCALE_BY_CHUNK: data above 1 are scaled by the maximum value, with the maximum
        value calculated over each slice, where slice shapes are determined by the group_by
        parameters

    See Also
    --------
    starfish.types.Axes

    """

    def __init__(
        self, dims: Iterable[Union[Axes, str]], func: Union[str, Callable] = 'max',
        clip_method: Union[str, Clip] = Clip.CLIP
    ) -> None:

        self.dims = dims
        self.clip_method = clip_method

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

        if self.clip_method == Clip.CLIP:
            reduced = preserve_float_range(reduced, rescale=False)
        else:
            reduced = preserve_float_range(reduced, rescale=True)

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
    @click.option(
        "--clip-method", default=Clip.CLIP, type=Clip,
        help="method to constrain data to [0,1]. options: 'clip', 'scale_by_image', "
             "'scale_by_chunk'")
    @click.pass_context
    def _cli(ctx, dims, func, clip_method):
        ctx.obj["component"]._cli_run(ctx, Reduce(dims, func, clip_method))
