from functools import partial
from typing import Optional, Union

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, Clip
from starfish.core.util import click
from ._base import FilterAlgorithmBase


class LinearUnmixing(FilterAlgorithmBase):
    """
    LinearUnmixing enables the user to correct fluorescent bleed by subtracting fractions of the
    intensities of other channels from each channel in the ImageStack.

    Examples
    --------
    The following example provides a coefficient matrix that corrects for spectral
    mixing in a 3-channel experiment.
    Channel 0 contains a mixture of itself plus 50% of the intensity of
    channel 2. Channel 1 has no mixing with other channels. Channel 3
    consists of itself plus 10% of the intensity of both channels 0 and 1.

    >>> import numpy as np
    >>> coeff_mat = np.ndarray([
    ...     [1,    0,  -0.1]
    ...     [0,    1,  -0.1]
    ...     [-0.5, 0,  1   ]
    ... ])

    The end result of this unmixing will be that 50% of channel 2 will be subtracted from
    channel 0, channel 1 will not be changed, and 10% of channels 0 and 1 will be subtracted
    from channel 2.

    Parameters
    ----------
    coeff_mat : np.ndarray
        matrix of the linear unmixing coefficients. Should take the form: B = AX, where B are
        the unmixed values, A is coeff_mat and X are the observed values. coeff_mat has shape
        (n_ch, n_ch), and poses each channel (column) as a combination of other columns (rows).
    clip_method : Union[str, Clip]
        (Default Clip.CLIP) Controls the way that data are scaled to retain skimage dtype
        requirements that float data fall in [0, 1].
        Clip.CLIP: data above 1 are set to 1, and below 0 are set to 0
        Clip.SCALE_BY_IMAGE: data above 1 are scaled by the maximum value, with the maximum
        value calculated over the entire ImageStack
        Clip.SCALE_BY_CHUNK: data above 1 are scaled by the maximum value, with the maximum
        value calculated over each slice, where slice shapes are determined by the group_by
        parameters

    """

    def __init__(
        self, coeff_mat: np.ndarray, clip_method: Union[str, Clip] = Clip.SCALE_BY_IMAGE,
    ) -> None:

        self.coeff_mat = coeff_mat
        self.clip_method = clip_method

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
            the observed values. coeff_mat has shape (n_ch, n_ch), and poses
            each channel (column) as a combination of other columns (rows).

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
        broadcast_image = image.values[..., None] * broadcast_coeff

        # collapse the unmixed result
        unmixed_image = np.sum(broadcast_image, axis=0).transpose([2, 0, 1])

        return unmixed_image

    def run(
            self,
            stack: ImageStack,
            in_place: bool=False,
            verbose: bool=False,
            n_processes: Optional[int]=None,
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
            Number of parallel processes to devote to applying the filter. If None, defaults to
            the result of os.cpu_count(). (default None)

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
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes,
            clip_method=self.clip_method,
        )
        return result

    @staticmethod
    @click.command("LinearUnmixing")
    @click.option(
        "--coeff_mat", required=True, type=np.ndarray, help="linear unmixing coefficients")
    @click.option(
        "--clip-method", default='scale_by_image',
        type=click.Choice(['clip', 'scale_by_image', 'scale_by_chunk']),
        help="method to constrain data to [0,1]")
    @click.pass_context
    def _cli(ctx, coeff_mat, clip_method):
        ctx.obj["component"]._cli_run(ctx, LinearUnmixing(coeff_mat, clip_method))
