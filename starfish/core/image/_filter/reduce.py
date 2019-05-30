from typing import (
    Callable,
    Iterable,
    MutableMapping,
    Sequence,
    Union
)

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, Clip, Coordinates, Number
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
        If a string is provided, it should correspond to a numpy function that
        matches the form specified above
        (i.e., function is resolved: func = getattr(np, func)).
        Some common examples below:
        amax: maximum intensity projection (applies numpy.amax)
        max: maximum intensity projection (this is an alias for amax and applies numpy.amax)
        mean: take the mean across the dim(s) (applies numpy.mean)
        sum: sum across the dim(s) (applies numpy.sum)
    clip_method : Clip
        (Default Clip.CLIP) Controls the way that data are scaled to retain skimage dtype
        requirements that float data fall in [0, 1].
        Clip.CLIP: data above 1 are set to 1, and below 0 are set to 0
        Clip.SCALE_BY_IMAGE: data above 1 are scaled by the maximum value, with the maximum
        value calculated over the entire ImageStack

    See Also
    --------
    starfish.types.Axes

    """

    def __init__(
        self, dims: Iterable[Union[Axes, str]], func: Union[str, Callable] = 'max',
        clip_method: Clip = Clip.CLIP
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
            *args,
    ) -> ImageStack:
        """Performs the dimension reduction with the specifed function

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack. Otherwise return the
            original stack.

        """

        # Apply the reducing function
        reduced = stack._data.reduce(self.func, dim=[Axes(dim).value for dim in self.dims])

        # Add the reduced dims back and align with the original stack
        reduced = reduced.expand_dims(tuple(Axes(dim).value for dim in self.dims))
        reduced = reduced.transpose(*stack.xarray.dims)

        if self.clip_method == Clip.CLIP:
            reduced = preserve_float_range(reduced, rescale=False)
        else:
            reduced = preserve_float_range(reduced, rescale=True)

        # Update the physical coordinates
        physical_coords: MutableMapping[Coordinates, Sequence[Number]] = {}
        for axis, coord in (
                (Axes.X, Coordinates.X),
                (Axes.Y, Coordinates.Y),
                (Axes.ZPLANE, Coordinates.Z)):
            if axis in self.dims:
                # this axis was projected out of existence.
                assert coord.value not in reduced.coords
                physical_coords[coord] = [np.average(stack._data.coords[coord.value])]
            else:
                physical_coords[coord] = reduced.coords[coord.value]
        reduced_stack = ImageStack.from_numpy(reduced.values, coordinates=physical_coords)

        return reduced_stack

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
