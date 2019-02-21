from copy import deepcopy

import numpy as np
import xarray as xr

from starfish.imagestack.imagestack import ImageStack
from starfish.util import click
from ._base import FilterAlgorithmBase
from .util import preserve_float_range


class ElementWiseMultiply(FilterAlgorithmBase):

    def __init__(self, mult_array: xr.core.dataarray.DataArray) -> None:
        """Perform elementwise multiplication on the image tensor. This is useful for
        performing operations such as image normalization or field flatness correction

        Parameters
        ----------
        mult_mat : xr.DataArray
            the image is element-wise multiplied with this array

        """
        self.mult_array = mult_array

    _DEFAULT_TESTING_PARAMETERS = {
        "mult_array": xr.DataArray(
            np.array([[[[[1]]], [[[0.5]]]]]),
            dims=('r', 'c', 'z', 'y', 'x')
        )
    }

    def run(
            self, stack: ImageStack, in_place: bool=False, verbose=None, n_processes=None
    ) -> ImageStack:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : None
            Not used. Elementwise multiply carries out a single vectorized multiplication that
            cannot provide a status bar. Included for consistency with Filter API.
        n_processes : None
            Not used. Elementwise multiplication scales slowly with additional processes due to the
            efficiency of vectorization on a single process. Included for consistency with Filter
            API. All computation happens on the main process.

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """

        # Align the axes of the multipliers with ImageStack
        mult_array_aligned: np.ndarray = self.mult_array.transpose(*stack.xarray.dims).values
        if not in_place:
            stack = deepcopy(stack)

        # stack._data contains the xarray
        stack._data *= mult_array_aligned
        stack._data = preserve_float_range(stack._data)
        return stack

    @staticmethod
    @click.command("ElementWiseMultiply")
    @click.option(
        "--mult-array", required=True, type=np.ndarray, help="matrix to multiply with the image")
    def _cli(ctx, mult_array):
        ctx.obj["component"]._cli_run(ctx, ElementWiseMultiply(mult_array))
