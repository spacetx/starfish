from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from skimage.feature import blob_log

from starfish.intensity_table import IntensityTable
from starfish.stack import ImageStack
from starfish.types import Features, Indices, Number, SpotAttributes
from starfish.util.argparse import FsExistsType
from ._base import SpotFinderAlgorithmBase
from .detect import detect_spots, measure_spot_intensity


class GaussianSpotDetector(SpotFinderAlgorithmBase):

    def __init__(
            self,
            min_sigma: Number,
            max_sigma: Number,
            num_sigma: int,
            threshold: Number,
            overlap: float=0.5,
            measurement_type='max',
            is_volume: bool=True, **kwargs
    ) -> None:
        """Multi-dimensional gaussian spot detector

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

        Notes
        -----
        # TODO ambrosejcarr: revisit after changing dtype assumptions of library to float in [0, 1]
        This spot detector is very sensitive to the threshold that is selected, and the threshold
        is defined as an absolute value -- therefore it must be adjusted depending on the datatype
        of the passed image.

        See Also
        --------
        http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log

        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.is_volume = is_volume
        self.measurement_function = self._get_measurement_function(measurement_type)

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:
        """
        Find spots using a gaussian blob finding algorithm

        Parameters
        ----------
        data_image : Union[np.ndarray, xr.DataArray]
            ImageStack containing blobs to be detected

        Returns
        -------
        SpotAttributes :
            DataFrame of metadata containing the coordinates, intensity and radius of each spot

        """

        fitted_blobs_array: np.ndarray = blob_log(
            data_image,
            self.min_sigma,
            self.max_sigma,
            self.num_sigma,
            self.threshold,
            self.overlap
        )

        # create the SpotAttributes Table
        columns = [Indices.Z.value, Indices.Y.value, Indices.X.value, Features.SPOT_RADIUS]
        fitted_blobs = pd.DataFrame(data=fitted_blobs_array, columns=columns)

        # convert standard deviation of gaussian kernel used to identify spot to radius of spot
        converted_radius = np.round(fitted_blobs[Features.SPOT_RADIUS] * np.sqrt(3))
        fitted_blobs[Features.SPOT_RADIUS] = converted_radius

        # convert the array to int so it can be used to index
        rounded_blobs: pd.DataFrame = fitted_blobs.astype(int)

        rounded_blobs['intensity'] = measure_spot_intensity(
            data_image, rounded_blobs, self.measurement_function)
        rounded_blobs['spot_id'] = np.arange(rounded_blobs.shape[0])

        return SpotAttributes(rounded_blobs)

    def run(
            self,
            data_stack: ImageStack,
            blobs_image: Optional[Union[np.ndarray, xr.DataArray]]=None,
            reference_image_from_max_projection: bool=False,
    ) -> IntensityTable:
        """find spots in an ImageStack

        Parameters
        ----------
        data_stack : ImageStack
            stack containing spots to find
        blobs_image : Union[np.ndarray, xr.DataArray]
        reference_image_from_max_projection : bool
            if True, compute a reference image from the maximum projection of the channels and
            z-planes

        Returns
        -------
        IntensityTable :
            3d tensor containing the intensity of spots across channels and imaging rounds

        """

        intensity_table = detect_spots(
            data_stack=data_stack,
            spot_finding_method=self.image_to_spots,
            reference_image=blobs_image,
            reference_image_from_max_projection=reference_image_from_max_projection,
            measurement_function=self.measurement_function,
            radius_is_gyration=False,
        )

        return intensity_table

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--blobs-stack", type=FsExistsType(), required=True)
        group_parser.add_argument(
            "--min-sigma", default=4, type=int, help="Minimum spot size (in standard deviation)")
        group_parser.add_argument(
            "--max-sigma", default=6, type=int, help="Maximum spot size (in standard deviation)")
        group_parser.add_argument(
            "--num-sigma", default=20, type=int, help="Number of sigmas to try")
        group_parser.add_argument("--threshold", default=.01, type=float, help="Dots threshold")
        group_parser.add_argument(
            "--overlap", default=0.5, type=float,
            help="dots with overlap of greater than this fraction are combined")
        group_parser.add_argument(
            "--show", default=False, action='store_true', help="display results visually")
