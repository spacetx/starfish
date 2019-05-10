from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from skimage.feature import blob_dog, blob_doh, blob_log

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, Number, SpotAttributes
from starfish.core.util import click
from ._base import DetectSpotsAlgorithmBase
from .detect import detect_spots, measure_spot_intensity

blob_detectors = {
    'blob_dog': blob_dog,
    'blob_doh': blob_doh,
    'blob_log': blob_log
}


class BlobDetector(DetectSpotsAlgorithmBase):
    """
    Multi-dimensional gaussian spot detector

    This method is a wrapper for skimage.feature.blob_log

    Parameters
    ----------
    min_sigma : float
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    num_sigma : int
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float [0, 1]
        If two spots have more than this fraction of overlap, the spots are combined
        (default = 0.5)
    measurement_type : str ['max', 'mean']
        name of the function used to calculate the intensity for each identified spot area
    detector_method: str ['blob_dog', 'blob_doh', 'blob_log']
        name of the type of detection method used from skimage.feature, default: blob_log

    Notes
    -----
    see also: http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html

    """

    def __init__(
            self,
            min_sigma: Number,
            max_sigma: Number,
            num_sigma: int,
            threshold: Number,
            overlap: float = 0.5,
            measurement_type='max',
            is_volume: bool = True,
            detector_method: str = 'blob_log'
    ) -> None:

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.is_volume = is_volume
        self.measurement_function = self._get_measurement_function(measurement_type)
        try:
            self.detector_method = blob_detectors[detector_method]
        except ValueError:
            raise ValueError("Detector method must be one of {blob_log, blob_dog, blob_doh}")

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:
        """
        Find spots using a gaussian blob finding algorithm

        Parameters
        ----------
        data_image : Union[np.ndarray, xr.DataArray]
            image containing spots to be detected

        Returns
        -------
        SpotAttributes :
            DataFrame of metadata containing the coordinates, intensity and radius of each spot

        """

        fitted_blobs_array: np.ndarray = self.detector_method(
            data_image,
            self.min_sigma,
            self.max_sigma,
            self.num_sigma,
            self.threshold,
            self.overlap
        )

        if fitted_blobs_array.shape[0] == 0:
            return SpotAttributes.empty(extra_fields=['intensity', 'spot_id'])

        # create the SpotAttributes Table
        columns = [Axes.ZPLANE.value, Axes.Y.value, Axes.X.value, Features.SPOT_RADIUS]
        fitted_blobs = pd.DataFrame(data=fitted_blobs_array, columns=columns)

        # convert standard deviation of gaussian kernel used to identify spot to radius of spot
        converted_radius = np.round(fitted_blobs[Features.SPOT_RADIUS] * np.sqrt(3))
        fitted_blobs[Features.SPOT_RADIUS] = converted_radius

        # convert the array to int so it can be used to index
        spots = SpotAttributes(fitted_blobs)

        spots.data['intensity'] = measure_spot_intensity(
            data_image, spots, self.measurement_function)
        spots.data['spot_id'] = np.arange(spots.data.shape[0])

        return spots

    def run(
            self,
            primary_image: ImageStack,
            blobs_image: Optional[ImageStack] = None,
            blobs_axes: Optional[Tuple[Axes, ...]] = None,
            n_processes: Optional[int] = None,
            *args,
    ) -> IntensityTable:
        """
        Find spots.

        Parameters
        ----------
        primary_image : ImageStack
            ImageStack where we find the spots in.
        blobs_image : Optional[ImageStack]
            If provided, spots will be found in the blobs image, and intensities will be measured
            across rounds and channels. Otherwise, spots are measured independently for each channel
            and round.
        blobs_axes : Optional[Tuple[Axes, ...]]
            If blobs_image is provided, blobs_axes must be provided as well.  blobs_axes represents
            the axes across which the blobs image is max projected before spot detection is done.
        n_processes : Optional[int] = None,
            Number of processes to devote to spot finding.
        """

        intensity_table = detect_spots(
            data_stack=primary_image,
            spot_finding_method=self.image_to_spots,
            reference_image=blobs_image,
            reference_image_max_projection_axes=blobs_axes,
            measurement_function=self.measurement_function,
            n_processes=n_processes,
            radius_is_gyration=False)

        return intensity_table

    @staticmethod
    @click.command("BlobDetector")
    @click.option(
        "--min-sigma", default=4, type=int, help="Minimum spot size (in standard deviation)")
    @click.option(
        "--max-sigma", default=6, type=int, help="Maximum spot size (in standard deviation)")
    @click.option(
        "--num-sigma", default=20, type=int, help="Number of sigmas to try")
    @click.option(
        "--threshold", default=.01, type=float, help="Dots threshold")
    @click.option(
        "--overlap", default=0.5, type=float,
        help="dots with overlap of greater than this fraction are combined")
    @click.option(
        "--show", default=False, is_flag=True, help="display results visually")
    @click.option(
        "--detector_method", default='blob_log',
        help="str ['blob_dog', 'blob_doh', 'blob_log'] name of the type of "
             "detection method used from skimage.feature. Default: blob_log"
    )
    @click.pass_context
    def _cli(ctx, min_sigma, max_sigma, num_sigma, threshold, overlap, show, detector_method):
        instance = BlobDetector(min_sigma, max_sigma, num_sigma, threshold, overlap,
                                detector_method=detector_method)
        #  FIXME: measurement_type, is_volume missing as options; show missing as ctor args
        ctx.obj["component"]._cli_run(ctx, instance)
