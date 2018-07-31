from itertools import product
from numbers import Number
from typing import Tuple

import numpy as np
import pandas as pd
from skimage.feature import blob_log

from starfish.constants import Indices, Features
from starfish.image import ImageStack
from starfish.munge import dataframe_to_multiindex
from starfish.intensity_table import IntensityTable
from starfish.util.argparse import FsExistsType
from ._base import SpotFinderAlgorithmBase


class GaussianSpotDetector(SpotFinderAlgorithmBase):

    def __init__(
            self, min_sigma, max_sigma, num_sigma, threshold,
            blobs_stack, overlap=0.5, measurement_type='max', is_volume: bool=True, **kwargs
    ) -> None:
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
            If two spots have more than this fraction of overlap, the spots are combined (default = 0.5)
        blobs_stack : Union[ImageStack, str]
            ImageStack or the path or URL that references the ImageStack that contains the blobs.
        measurement_type : str ['max', 'mean']
            name of the function used to calculate the intensity for each identified spot area

        Notes
        -----
        This spot detector is very sensitive to the threshold that is selected, and the threshold is defined as an
        absolute value -- therefore it must be adjusted depending on the datatype of the passed image.


        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.is_volume = is_volume
        if isinstance(blobs_stack, ImageStack):
            self.blobs_stack = blobs_stack
        else:
            self.blobs_stack = ImageStack.from_path_or_url(blobs_stack)
        self.blobs_image: np.ndarray = self.blobs_stack.max_proj(Indices.ROUND, Indices.CH)

        try:
            self.measurement_function = getattr(np, measurement_type)
        except AttributeError:
            raise ValueError(
                f'measurement_type must be a numpy reduce function such as "max" or "mean". {measurement_type} '
                f'not found.')

    @staticmethod
    def _measure_blob_intensity(image, blobs, measurement_function) -> pd.Series:

        def fn(row: pd.Series) -> Number:
            row = row.astype(int)
            result = measurement_function(
                image[
                    row['z_min']:row['z_max'],
                    row['y_min']:row['y_max'],
                    row['x_min']:row['x_max']
                ]
            )
            return result

        return blobs.apply(
            fn,
            axis=1
        )

    def _measure_spot_intensities(
            self, stack: ImageStack, spot_attributes: pd.DataFrame
    ) -> IntensityTable:

        n_ch = stack.shape[Indices.CH]
        n_round = stack.shape[Indices.ROUND]
        spot_attribute_index = dataframe_to_multiindex(spot_attributes)
        image_shape: Tuple[int, int, int] = stack.raw_shape[2:]
        intensity_table = IntensityTable.empty_intensity_table(
            spot_attribute_index, n_ch, n_round, image_shape)

        indices = product(range(n_ch), range(n_round))
        for c, h in indices:
            image, _ = stack.get_slice({Indices.CH: c, Indices.ROUND: h})
            blob_intensities: pd.Series = self._measure_blob_intensity(
                image, spot_attributes, self.measurement_function)
            intensity_table[:, c, h] = blob_intensities

        return intensity_table

    def _find_spot_locations(self) -> pd.DataFrame:
        fitted_blobs_array: np.ndarray = blob_log(
            self.blobs_image, self.min_sigma, self.max_sigma, self.num_sigma, self.threshold,
            self.overlap)

        if fitted_blobs_array.shape[0] == 0:
            raise ValueError('No spots detected with provided parameters')

        columns = [Features.Z, Features.Y, Features.X, Features.SPOT_RADIUS]
        fitted_blobs = pd.DataFrame(data=fitted_blobs_array, columns=columns)

        # convert standard deviation of gaussian kernel used to identify spot to radius of spot
        converted_radius = np.round(fitted_blobs[Features.SPOT_RADIUS] * np.sqrt(3))
        fitted_blobs[Features.SPOT_RADIUS] = converted_radius

        # convert the array to int so it can be used to index
        fitted_blobs = fitted_blobs.astype(int)

        for v, max_size in zip(['z', 'y', 'x'], self.blobs_image.shape):
            fitted_blobs[f'{v}_min'] = np.clip(
                fitted_blobs[v] - fitted_blobs[Features.SPOT_RADIUS], 0, None)
            fitted_blobs[f'{v}_max'] = np.clip(
                fitted_blobs[v] + fitted_blobs[Features.SPOT_RADIUS], None, max_size)

        fitted_blobs['intensity'] = self._measure_blob_intensity(
            self.blobs_image, fitted_blobs, self.measurement_function)
        fitted_blobs['spot_id'] = np.arange(fitted_blobs.shape[0])

        return fitted_blobs

    def find(self, image_stack: ImageStack) -> IntensityTable:
        """find spots

        Parameters
        ----------
        image_stack : ImageStack
            stack containing spots to find

        Returns
        -------
        IntensityTable :
            3d tensor containing the intensity of spots across channels and imaging rounds

        """
        spot_attributes = self._find_spot_locations()
        intensity_table = self._measure_spot_intensities(image_stack, spot_attributes)
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
