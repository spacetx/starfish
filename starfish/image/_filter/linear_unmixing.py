from functools import partial
from typing import Optional

import numpy as np

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes
from starfish.util import click
from ._base import FilterAlgorithmBase
from .util import preserve_float_range


class LinearUnmixing(FilterAlgorithmBase):

    def __init__(self, coeff_mat: np.ndarray) -> None:
        """Image scaling filter

        Parameters
        ----------
        coeff_mat : np.ndarray
            matrix of the linear unmixing coefficients. Should take the form:
            B = AX, where B are the unmixed values, A is coeff_mat and X are
            the observed values.

        """
        self.coeff_mat = coeff_mat

    _DEFAULT_TESTING_PARAMETERS = {"coeff_mat": np.array([[1, -0.25], [-0.25, 1]])}

    @staticmethod
    def _unmix(image: np.ndarray, coeff_mat: np.ndarray) -> np.ndarray:
        """Clip values of img below and above percentiles p_min and p_max

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
        # Preallocate the new image in the same shape
        unmixed_image = np.zeros(image.shape)

        # Due to the grouping in run(), image is shape [chan, x, y]
        n_channels = image.shape[0]

        for c in range(n_channels):
            # Get the coefficients
            coeffs = np.zeros((n_channels, 1, 1))
            coeffs[:, 0, 0] = coeff_mat[c]

            coeff_im = np.multiply(image, coeffs)

            unmixed_image[c, ...] = np.sum(coeff_im, axis=0)

        unmixed_image = preserve_float_range(unmixed_image)

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
    @click.pass_context
    def _cli(ctx, coeff_mat):
        ctx.obj["component"]._cli_run(ctx, LinearUnmixing(coeff_mat))
