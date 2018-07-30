from typing import Callable, Sequence

import numpy as np
import pandas as pd
from skimage.feature import blob_log

from starfish.constants import Indices, Features
from starfish.image import ImageStack
from starfish.intensity_table import IntensityTable
from starfish.pipeline.features.spot_attributes import SpotAttributes
from starfish.pipeline.features.spots.detector.detect import (
    measure_spot_intensity,
    detect_spots,
)
from starfish.util.argparse import FsExistsType
from starfish.typing import Number
from ._base import SpotFinderAlgorithmBase


class GaussianSpotDetector(SpotFinderAlgorithmBase):

    def __init__(
            self, min_sigma: Number, max_sigma: Number, num_sigma: int, threshold: Number,
            blobs_stack: ImageStack, overlap=0.5, measurement_type='max', is_volume: bool=True,
            **kwargs) \
            -> None:
        """Multi-dimensional gaussian spot detector

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
        blobs_stack : Union[ImageStack, str]
            ImageStack or the path or URL that references the ImageStack that contains the blobs.
        measurement_type : str ['max', 'mean']
            name of the function used to calculate the intensity for each identified spot area

        Notes
        -----
        This spot detector is very sensitive to the threshold that is selected, and the threshold
        is defined as an absolute value -- therefore it must be adjusted depending on the datatype
        of the passed image.


        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.is_volume = is_volume
        if isinstance(blobs_stack, ImageStack):
            self.blobs_stack = blobs_stack
        elif isinstance(blobs_stack, str):
            self.blobs_stack = ImageStack.from_path_or_url(blobs_stack)
        else:
            raise TypeError(f"blobs_stack must be a string url pointing to an experiment.json file "
                            f"or an ImageStack, not {type(blobs_stack)}.")
        self.blobs_image: np.ndarray = self.blobs_stack.max_proj(Indices.ROUND, Indices.CH)

        try:
            self.measurement_function = getattr(np, measurement_type)
        except AttributeError:
            raise ValueError(
                f'measurement_type must be a numpy reduce function such as "max" or "mean". {measurement_type} '
                f'not found.')

    def find(self, image_stack: ImageStack) -> IntensityTable:
        """find spots in an ImageStack

        Parameters
        ----------
        image_stack : ImageStack
            stack containing spots to find

        Returns
        -------
        IntensityTable :
            3d tensor containing the intensity of spots across channels and imaging rounds

        """
        spot_finding_kwargs = {
            'min_sigma': self.min_sigma,
            'max_sigma': self.max_sigma,
            'num_sigma': self.num_sigma,
            'threshold': self.threshold,
            'overlap': self.overlap,
            'measurement_function': self.measurement_function,
        }

        intensity_table = detect_spots(
            data_image=image_stack,
            spot_finding_method=gaussian_spot_detector,
            spot_finding_kwargs=spot_finding_kwargs,
            reference_image=self.blobs_image,  # todo should be xarray with tony's PR
            measurement_function=self.measurement_function
        )

        return intensity_table

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--blobs-stack", type=FsExistsType(), required=True)
        group_parser.add_argument(
            "--min-sigma", default=4, type=int, help="Minimum spot size (in standard deviation)")
        group_parser.add_argument(
            "--max-sigma", default=6, type=int, help="Maximum spot size (in standard deviation)")
        group_parser.add_argument("--num-sigma", default=20, type=int, help="Number of sigmas to try")
        group_parser.add_argument("--threshold", default=.01, type=float, help="Dots threshold")
        group_parser.add_argument(
            "--overlap", default=0.5, type=float, help="dots with overlap of greater than this fraction are combined")
        group_parser.add_argument(
            "--show", default=False, action='store_true', help="display results visually")


# TODO ambrosejcarr: make this return IntensityTable instead of SpotAttributes
def gaussian_spot_detector(
        data_image: ImageStack, min_sigma: Number, max_sigma: Number, num_sigma: int,
        threshold: Number, overlap=0.5, measurement_function: Callable[[Sequence], Number]=np.max) \
        -> SpotAttributes:
    """
    Find gaussian blobs in an data image

    Parameters
    ----------
    data_image : ImageStack
        ImageStack containing blobs to be detected
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
        If two spots have more than this fraction of overlap, the spots are combined (default = 0.5)
    measurement_function : Callable
        The function used to calculate the intensity for each identified spot area

    Returns
    -------
    SpotAttributes :
        DataFrame of metadata containing the coordinates, intensity and radius of each spot

    """

    fitted_blobs_array: np.ndarray = blob_log(
        data_image,
        min_sigma,
        max_sigma,
        num_sigma,
        threshold,
        overlap
    )

    # TODO this needs to be a warning, there are codebooks that could trigger this
    if fitted_blobs_array.shape[0] == 0:
        raise ValueError('No spots detected with provided parameters')

    # create the SpotAttributes Table
    columns = [Features.Z, Features.Y, Features.X, Features.SPOT_RADIUS]
    fitted_blobs = pd.DataFrame(data=fitted_blobs_array, columns=columns)

    # convert standard deviation of gaussian kernel used to identify spot to radius of spot
    converted_radius = np.round(fitted_blobs[Features.SPOT_RADIUS] * np.sqrt(3))
    fitted_blobs[Features.SPOT_RADIUS] = converted_radius

    # convert the array to int so it can be used to index
    rounded_blobs: pd.DataFrame = fitted_blobs.astype(int)

    for v, max_size in zip(['z', 'y', 'x'], data_image.shape):
        rounded_blobs[f'{v}_min'] = np.clip(
            rounded_blobs[v] - rounded_blobs[Features.SPOT_RADIUS], 0, None)
        rounded_blobs[f'{v}_max'] = np.clip(
            rounded_blobs[v] + rounded_blobs[Features.SPOT_RADIUS], None, max_size)

    rounded_blobs['intensity'] = measure_spot_intensity(
        data_image, rounded_blobs, measurement_function)
    rounded_blobs['spot_id'] = np.arange(rounded_blobs.shape[0])

    return SpotAttributes(rounded_blobs)
