from functools import partial
from typing import Optional

import numpy as np

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes
from starfish.util import click
from ._base import FilterAlgorithmBase
from .util import preserve_float_range


class ElementWiseMult(FilterAlgorithmBase):

    def __init__(self, mult_mat: np.ndarray) -> None:
        """Image scaling filter

        Parameters
        ----------
        mult_mat : np.ndarray
            each image in the stack is scaled by this percentile.

        """
        self.mult_mat = mult_mat

    _DEFAULT_TESTING_PARAMETERS = {"mult_mat": np.array([[[[[1]]], [[[0.5]]]]])}

    @staticmethod
    def _mult(image: np.ndarray, mult_mat: np.ndarray) -> np.ndarray:
        """Perform elementwise multiplication on the image tensor. This is useful for
        performing operations such as image normalization or field flatness correction

        Parameters
        ----------
        image : np.ndarray
            image to be scaled

        mult_mat : np.ndarray
            each image in the stack is element-wise multiplied by this matrix.


        Returns
        -------
        np.ndarray :
          Numpy array of same shape as img

        """

        # Get the axes to squeeze and squeeze the mult_mat while
        # preserving the X, Y axes
        to_squeeze = np.isin(mult_mat.shape[0:3], 1)
        squeezable_axes = np.array([0, 1, 2])
        axes_to_squeeze = tuple(squeezable_axes[to_squeeze])

        squeezed_mult_mat = np.squeeze(mult_mat, axis=axes_to_squeeze)

        # Element-wise mult
        image = np.multiply(image, squeezed_mult_mat)

        image = preserve_float_range(image)

        return image

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
        # Build the grouping set
        group_by = set()

        if self.mult_mat.shape[0] == 1:
            group_by.add(Axes.ROUND)
        if self.mult_mat.shape[1] == 1:
            group_by.add(Axes.CH)
        if self.mult_mat.shape[2] == 1:
            group_by.add(Axes.ZPLANE)

        clip = partial(self._mult, mult_mat=self.mult_mat)
        result = stack.apply(
            clip,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result

    @staticmethod
    @click.command("ElementWiseMult")
    @click.option(
        "--mult-mat", required=True, type=np.ndarray, help="matrix to multiply with the image")
    def _cli(ctx, mult_mat):
        ctx.obj["component"]._cli_run(ctx, ElementWiseMult(mult_mat))
