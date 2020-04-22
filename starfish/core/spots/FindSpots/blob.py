from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_doh, blob_log

from starfish.core.image.Filter.util import determine_axes_to_group_by
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.spots.FindSpots import spot_finding_utils
from starfish.core.types import (
    Axes,
    Features,
    Number,
    PerImageSliceSpotResults,
    SpotAttributes,
    SpotFindingResults,
)
from ._base import FindSpotsAlgorithm

blob_detectors = {
    'blob_dog': blob_dog,
    'blob_doh': blob_doh,
    'blob_log': blob_log
}


class BlobDetector(FindSpotsAlgorithm):
    """
    Multi-dimensional gaussian spot detector

    This method is a wrapper for :py:func:`skimage.feature.blob_log`

    Parameters
    ----------
    min_sigma : Number
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : Number
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    num_sigma : int
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float
        The absolute lower bound for scale space maxima. Local maxima smaller
        than threshold are ignored. Reduce this to detect blobs with less
        intensities.
    is_volume: bool
        If True, pass 3d volumes (x, y, z) to func, else pass 2d tiles (x, y) to func. (default:
        True)
    overlap : float [0, 1]
        If two spots have more than this fraction of overlap, the spots are combined
        (default: 0.5)
    measurement_type : str ['max', 'mean']
        name of the function used to calculate the intensity for each identified spot area
        (default: max)
    detector_method: str ['blob_dog', 'blob_doh', 'blob_log']
        name of the type of detection method used from :py:mod:`~skimage.feature`
        (default: blob_log)

    Notes
    -----
    See also: :doc:`skimage:auto_examples/features_detection/plot_blob`

    """

    def __init__(
            self,
            min_sigma: Union[Number, Tuple[Number, ...]],
            max_sigma: Union[Number, Tuple[Number, ...]],
            num_sigma: int,
            threshold: Number,
            overlap: float = 0.5,
            measurement_type='max',
            is_volume: bool = True,
            detector_method: str = 'blob_log',
            exclude_border: Optional[int] = None,
    ) -> None:

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.is_volume = is_volume
        self.measurement_function = self._get_measurement_function(measurement_type)
        self.exclude_border = exclude_border
        try:
            self.detector_method = blob_detectors[detector_method]
        except ValueError:
            raise ValueError("Detector method must be one of {blob_log, blob_dog, blob_doh}")

    def image_to_spots(
            self, data_image: np.ndarray,
    ) -> PerImageSliceSpotResults:
        """
        Find spots using a gaussian blob finding algorithm

        Parameters
        ----------
        data_image : np.ndarray
            image containing spots to be detected

        Returns
        -------
        PerImageSpotResults :
            includes a SpotAttributes DataFrame of metadata containing the coordinates, intensity
            and radius of each spot, as well as any extra information collected during spot finding.

        """

        spot_finding_args = {
            "min_sigma": self.min_sigma,
            "max_sigma": self.max_sigma,
            "threshold": self.threshold,
            "exclude_border": self.exclude_border,
            "overlap": self.overlap,
            "num_sigma": self.num_sigma
        }
        if self.detector_method is not blob_doh:
            spot_finding_args["exclude_border"] = self.exclude_border

        fitted_blobs_array: np.ndarray = self.detector_method(
            data_image,
            **spot_finding_args
        )

        if fitted_blobs_array.shape[0] == 0:
            empty_spot_attrs = SpotAttributes.empty(
                extra_fields=[Features.INTENSITY, Features.SPOT_ID])
            return PerImageSliceSpotResults(spot_attrs=empty_spot_attrs, extras=None)

        # measure intensities
        z_inds = fitted_blobs_array[:, 0].astype(int)
        y_inds = fitted_blobs_array[:, 1].astype(int)
        x_inds = fitted_blobs_array[:, 2].astype(int)
        radius = np.round(fitted_blobs_array[:, 3] * np.sqrt(3))
        data_image = np.asarray(data_image)
        intensities = data_image[tuple([z_inds, y_inds, x_inds])]

        # construct dataframe
        spot_data = pd.DataFrame(
            data={
                Features.INTENSITY: intensities,
                Axes.ZPLANE.value: z_inds,
                Axes.Y.value: y_inds,
                Axes.X.value: x_inds,
                Features.SPOT_RADIUS: radius,
            }
        )
        spots = SpotAttributes(spot_data)
        spots.data[Features.SPOT_ID] = np.arange(spots.data.shape[0])
        return PerImageSliceSpotResults(spot_attrs=spots, extras=None)

    def run(
            self,
            image_stack: ImageStack,
            reference_image: Optional[ImageStack] = None,
            n_processes: Optional[int] = None,
            *args,
    ) -> SpotFindingResults:
        """
        Find spots in the given ImageStack using a gaussian blob finding algorithm.
        If a reference image is provided the spots will be detected there then measured
        across all rounds and channels in the corresponding ImageStack. If a reference_image
        is not provided spots will be detected _independently_ in each channel. This assumes
        a non-multiplex imaging experiment, as only one (ch, round) will be measured for each spot.

        Parameters
        ----------
        image_stack : ImageStack
            ImageStack where we find the spots in.
        reference_image : Optional[ImageStack]
            (Optional) a reference image. If provided, spots will be found in this image, and then
            the locations that correspond to these spots will be measured across each channel.
        n_processes : Optional[int] = None,
            Number of processes to devote to spot finding.
        """
        spot_finding_method = partial(self.image_to_spots, *args)
        if reference_image:
            data_image = reference_image._squeezed_numpy(*{Axes.ROUND, Axes.CH})
            if self.detector_method is blob_doh and data_image.ndim > 2:
                raise ValueError("blob_doh only support 2d images")
            reference_spots = spot_finding_method(data_image)
            results = spot_finding_utils.measure_intensities_at_spot_locations_across_imagestack(
                data_image=image_stack,
                reference_spots=reference_spots,
                measurement_function=self.measurement_function)
        else:
            if self.detector_method is blob_doh and self.is_volume:
                raise ValueError("blob_doh only support 2d images")
            spot_attributes_list = image_stack.transform(
                func=spot_finding_method,
                group_by=determine_axes_to_group_by(self.is_volume),
                n_processes=n_processes
            )
            results = SpotFindingResults(imagestack_coords=image_stack.xarray.coords,
                                         log=image_stack.log,
                                         spot_attributes_list=spot_attributes_list)
        return results
