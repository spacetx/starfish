from typing import Tuple, Union, Optional, List

import numpy as np
import xarray as xr
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from sympy import Point, Line

from starfish import ImageStack, IntensityTable
from starfish.spots._detector._base import SpotFinderAlgorithmBase
from starfish.spots._detector.detect import detect_spots
from starfish.types import SpotAttributes, Number


class LocalMaxPeakFinder(SpotFinderAlgorithmBase):
    def __init__(
            self, min_distance, stringency, min_obj_area, max_obj_area, threshold=None
            , measurement_type: str = 'max', is_volume: bool = False,
            verbose=False, **kwargs) -> None:

        self.min_distance = min_distance
        self.stringency = stringency
        self.min_obj_area = min_obj_area
        self.max_obj_area = max_obj_area

        if threshold is None:
            self.threshold = self._compute_thresholds()
        else:
            self.threshold = threshold

        if is_volume:
            raise ValueError(
                'LocalMaxPeakFinder only works for 2D data, for 3D data, please use TrackpyLocalMaxPeakFinder')

        self.measurement_function = self._get_measurement_function(measurement_type)

    def _compute_thresholds(self, img: np.ndarray) -> Tuple[List, List]:

        # thresholds to search over
        thresholds = np.linspace(img.min(), img.max(), num=100)

        # number of spots detected at each threshold
        spot_counts = []

        # where we stop our threshold search
        stop_threshold = None

        for threshold in thresholds:
            spots = peak_local_max(img,
                                   min_distance=self.min_distance,
                                   threshold_abs=threshold,
                                   exclude_border=False,
                                   indices=True,
                                   num_peaks=np.inf,
                                   footprint=None,
                                   labels=None
                                   )

            # stop spot finding when the number of detected spots falls below 3
            if len(spots) <= 3:
                stop_threshold = threshold
                break
            else:
                spot_counts.append(len(spots))

        if stop_threshold is None:
            stop_threshold = thresholds.max()

        # for some reason, np.where returns a tuple of nd.arrays,
        # hence the [0][0] indexing to get out a numerical value
        stop_index = np.where(thresholds == stop_threshold)[0][0]

        if len(thresholds > 1):
            thresholds = thresholds[:stop_index]
            spot_counts = spot_counts[:stop_index]

        return thresholds, spot_counts

    @staticmethod
    def _select_optimal_threshold(thresholds: List, spot_counts: List, stringency: Number) -> Tuple[Number, Number]:

        # calculate the gradient of the number of spots
        grad = np.gradient(spot_counts)
        optimal_threshold_index = np.argmin(grad)

        # only consider thresholds > than optimal threshold
        thresholds = thresholds[optimal_threshold_index:]
        grad = grad[optimal_threshold_index:]

        if thresholds.shape > (1,):

            spot_counts = spot_counts[optimal_threshold_index:]

            distances = []

            # Calculate the coords of the end points of the gradient
            p1 = Point(thresholds[0], grad[0])
            p2 = Point(thresholds[-1], grad[-1])

            # Create a line that join the points
            s = Line(p1, p2)
            allpoints = np.arange(0, len(thresholds))

            # Calculate the distance between all points and the line
            for p in allpoints:
                dst = s.distance(Point(thresholds[p], grad[p]))
                distances.append(dst.evalf())

            # Remove the end points from the lists
            thresholds = thresholds[1:-1]
            trimmed_distances = distances[1:-1]

            # Determine the coords of the selected Thr
            # Converted trimmed_distances to array because it crashed
            # on Sanger.
            if trimmed_distances:  # Most efficient way will be to consider the length of Thr list
                thr_idx = np.argmax(np.array(trimmed_distances))

                if thr_idx + stringency < len(thresholds):
                    selected_thr = thresholds[thr_idx + stringency]
                    thr_idx = thr_idx + stringency
                else:
                    selected_thr = thresholds[thr_idx]

            else:
                thr_idx = 0
                selected_thr = 0

        return thr_idx, selected_thr

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:

        if self.threshold is None:
            self.threshold = self.compute_threshold()

        spot_coords = peak_local_max(data_image,
                                     min_distance=self.min_distance,
                                     threshold_abs=self.threshold,
                                     exclude_border=False,
                                     indices=True,
                                     num_peaks=np.inf,
                                     footprint=None,
                                     labels=None)

        masked_image = data_image > self.threshold
        labels = label(masked_image)[0]
        props = regionprops(labels)

        for prop in props:
            if prop.area < self.min_obj_area or prop.area > self.max_obj_area:
                masked_image[prop.coords[:, 0], prop.coords[:, 1]] = 0

        labels = label(masked_image)[0]

    def compute_threhsold(self):
        return 0.1

    def run(
            self,
            data_stack: ImageStack,
            blobs_image: Optional[Union[np.ndarray, xr.DataArray]] = None,
            reference_image_from_max_projection: bool = False,
    ) -> IntensityTable:
        """
        Find spots.

        Parameters
        ----------
        data_stack : ImageStack
            Stack where we find the spots in.
        blobs_image : Union[np.ndarray, xr.DataArray]
            If provided, spots will be found in the blobs image, and intensities will be measured
            across hybs and channels. Otherwise, spots are measured independently for each channel
            and round.
        reference_image_from_max_projection : bool
            if True, compute a reference image from the maximum projection of the channels and
            z-planes

        """
        intensity_table = detect_spots(
            data_stack=data_stack,
            spot_finding_method=self.image_to_spots,
            reference_image=blobs_image,
            reference_image_from_max_projection=reference_image_from_max_projection,
            measurement_function=self.measurement_function,
            radius_is_gyration=True,
        )

        return intensity_table
