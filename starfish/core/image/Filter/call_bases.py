from functools import partial
from typing import Optional

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.types import Axes
from ._base import FilterAlgorithm


class CallBases(FilterAlgorithm):
    """
    The CallBases filter determines the nucleotide present in each pixel of each
    (round, channel). The pixel values in the resulting image are the base quality
    score. The base quality score is calculated as the intensity of the pixel
    divided by the L2 norm of the all channels for that pixel. The base
    call score as a range of (0, 1) with a value of 0  and 1 being no call and a
    perfect call, respectively.

    Parameters
    ----------
    intensity_threshold : float
        Minimum intensity a pixel must have to be called a base.
        Set to zero for no thresholding.
    quality_threshold : float
        Minimum base quality score a pixel must have to be called a base.
        Set to zero for no thresholding.
    """

    def __init__(
        self, intensity_threshold: float = 0, quality_threshold: float = 0
    ) -> None:

        self.intensity_threshold = intensity_threshold
        self.quality_threshold = quality_threshold

    _DEFAULT_TESTING_PARAMETERS = {"intensity_threshold": 0, "quality_threshold": 0}

    def _vector_norm(self, x: xr.DataArray, dim: Axes, ord=None):
        """
        Calculates the vector norm across a given dimension of an xarray.

        Parameters
        ----------
        x : xarray.DataArray
            xarray to calcuate the norm.
        dim : Axes
            The dimension of the xarray to perform the norm across
        ord : Optional[str]
            Order of the norm to be applied. Leave none for L2 norm.

            See numpy docs for details:
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html

        Returns
        -------
        result : xarray.DataArray
            The resulting xarray of norms

        """
        result = xr.apply_ufunc(
            np.linalg.norm, x, input_core_dims=[[dim.value]],
            kwargs={'ord': ord, 'axis': -1}
        )

        return result

    def _call_bases(
        self, image: xr.DataArray, intensity_threshold: float,
        quality_threshold: float
    ) -> xr.DataArray:
        """
        Determines the nucleotide present in each pixel of each
        (round, channel). The pixel values in the resulting image are the base quality
        score. The base quality score is calculated as the intensity of the pixel
        divided by the L2 norm of the all channels for that pixel. The base
        call score as a range of (0, 1) with a value of 0  and 1 being no call and a
        perfect call, respectively.

        Parameters
        ----------
        image : xr.DataArray
            Image for base calling.
            Should have the following dims: Axes.CH, Axes.X, Axes.Y
        intensity_threshold : float
            Minimum intensity a pixel must have to be called a base.
            Set to zero for no thresholding.
        quality_threshold : float
            Minimum intensity a pixel must have to be called a base.
            Set to zero for no thresholding.

        """
        # Get the maximum value for each round/z
        max_chan = image.argmax(dim=Axes.CH.value)
        max_values = image.max(dim=Axes.CH.value)

        # Get the norms for each pixel
        norms = self._vector_norm(x=image, dim=Axes.CH)

        # Calculate the base qualities
        base_qualities = max_values / norms

        # Filter the base call qualities
        base_qualities_filtered = xr.where(
            base_qualities < quality_threshold, 0, base_qualities
        )

        # Threshold the intensity values
        base_qualities_filtered = xr.where(
            max_values < intensity_threshold, 0, base_qualities_filtered
        )

        # Put the base calls in place
        base_calls = xr.full_like(other=image, fill_value=0)
        base_calls[max_chan] = base_qualities_filtered

        return base_calls

    def run(
            self,
            stack: ImageStack,
            in_place: bool = False,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args,
    ) -> Optional[ImageStack]:
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
            If in-place is False, return the results of filter as a new stack. Otherwise return the
            original stack.

        """

        group_by = {Axes.ROUND, Axes.ZPLANE}
        unmix = partial(
            self._call_bases, intensity_threshold=self.intensity_threshold,
            quality_threshold=self.quality_threshold
        )
        result = stack.apply(
            unmix,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result
