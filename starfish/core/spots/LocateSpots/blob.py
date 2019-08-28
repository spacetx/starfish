from functools import partial
from typing import Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from skimage.feature import blob_dog, blob_doh, blob_log

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.spots.LocateSpots import spot_finding_utils
from starfish.core.types import Axes, Features, Number, SpotAttributes
from ._base import LocateSpotsAlgorithmBase

blob_detectors = {
    'blob_dog': blob_dog,
    'blob_doh': blob_doh,
    'blob_log': blob_log
}


class BlobDetector(LocateSpotsAlgorithmBase):
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
            return SpotAttributes.empty(extra_fields=[Features.INTENSITY, Features.SPOT_ID])

        # create the SpotAttributes Table
        columns = [Axes.ZPLANE.value, Axes.Y.value, Axes.X.value, Features.SPOT_RADIUS]
        fitted_blobs = pd.DataFrame(data=fitted_blobs_array, columns=columns)

        # convert standard deviation of gaussian kernel used to identify spot to radius of spot
        converted_radius = np.round(fitted_blobs[Features.SPOT_RADIUS] * np.sqrt(3))
        fitted_blobs[Features.SPOT_RADIUS] = converted_radius

        # convert the array to int so it can be used to index
        spots = SpotAttributes(fitted_blobs)
        spots.data[Features.SPOT_ID] = np.arange(spots.data.shape[0])

        return spots

    def run(
            self,
            image_stack: ImageStack,
            reference_image: Optional[ImageStack] = None,
            n_processes: Optional[int] = None,
            *args,
    ) -> Dict[Tuple, SpotAttributes]:
        """
        Find spots.

        Parameters
        ----------
        image_stack : ImageStack
            ImageStack where we find the spots in.
        reference_image : xr.DataArray
            (Optional) a reference image. If provided, spots will be found in this image, and then
            the locations that correspond to these spots will be measured across each channel and round,
            filling in the values in the IntensityTable
        n_processes : Optional[int] = None,
            Number of processes to devote to spot finding.
        """

        spot_finding_method = partial(self.image_to_spots, *args)
        if reference_image:
            spot_attributes_list = reference_image.transform(
                func=spot_finding_method,
                group_by={Axes.ROUND, Axes.CH},
                n_processes=n_processes
            )
            # todo technically just need first one but this is kinda hacky
            reference_spots = spot_attributes_list[0][0]
            measured_spots = spot_finding_utils.measure_spot_intensities(image_stack, reference_spots, np.mean)
            return measured_spots
        else:
            spot_attributes_list = image_stack.transform(
                func=spot_finding_method,
                group_by={Axes.ROUND, Axes.CH},
                n_processes=n_processes
            )
         # todo will probs need a converter in new Datastructre from spot attributes list
        return spot_attributes_list
