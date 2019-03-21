from functools import partial
from typing import Optional

import numpy as np
import xarray as xr

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes
from starfish.util import click
from ._base import FilterAlgorithmBase


class LinearUnmixing(FilterAlgorithmBase):

    def __init__(self, coeff_mat: np.ndarray, clip_method: int=1) -> None:
        """Image scaling filter

        Parameters
        ----------
        coeff_mat : np.ndarray
            matrix of the linear unmixing coefficients. Should take the form:
            B = AX, where B are the unmixed values, A is coeff_mat and X are
            the observed values.
        clip_method : int
            (Default 1) Controls the way that data are scaled to retain skimage dtype
            requirements that float data fall in [0, 1].
            0: data above 1 are set to 1, and below 0 are set to 0
            1: data above 1 are scaled by the maximum value, with the maximum value calculated
               over the entire ImageStack
            2: data above 1 are scaled by the maximum value, with the maximum value calculated
               over each slice, where slice shapes are determined by the group_by parameters

        """
        self.coeff_mat = coeff_mat

    _DEFAULT_TESTING_PARAMETERS = {"coeff_mat": np.array([[1, -0.25], [-0.25, 1]])}

    @staticmethod
    def _unmix(image: xr.DataArray, coeff_mat: np.ndarray) -> np.ndarray:
        """Perform linear unmixing of channels

        Parameters
        ----------
        image : np.ndarray
            image to be scaled

        coeff_mat : np.ndarray
            matrix of the linear unmixing coefficients. Should take the form:
            B = AX, where B are the unmixed values, A is coeff_mat and X are
            the observed values. coeff_mat has shape (n_ch, n_ch).

        Returns
        -------
        np.ndarray :
          Numpy array of same shape as image

        """

        x = image.sizes[Axes.X.value]
        y = image.sizes[Axes.Y.value]
        c = image.sizes[Axes.CH.value]

        # broadcast each channel coefficient across x and y
        broadcast_coeff = np.tile(coeff_mat, reps=x * y).reshape(c, y, x, c)

        # multiply the image by each coefficient
        unmixed_image = np.sum(image.values[..., None] * broadcast_coeff, axis=-1)

        return unmixed_image

    def run(
            self, stack: ImageStack, in_place: bool=False, verbose: bool=False,
            n_processes: Optional[int]=None
    ) -> ImageStack:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        group_by = {Axes.ROUND, Axes.ZPLANE}
        unmix = partial(self._unmix, coeff_mat=self.coeff_mat)
        result = stack.apply(
            unmix,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result

    @staticmethod
    @click.command("LinearUnmixing")
    @click.option(
        "--coeff_mat", required=True, type=np.ndarray, help="linear unmixing coefficients")
    @click.option(
        "--clip-method", default=1, type=int,
        help="method to constrain data to [0,1]. 0: clip, 1: scale by max over whole image, "
             "2: scale by max per chunk")
    @click.pass_context
    def _cli(ctx, coeff_mat):
        ctx.obj["component"]._cli_run(ctx, LinearUnmixing(coeff_mat))
