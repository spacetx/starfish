from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from sympy import Line, Point
from tqdm import tqdm

from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table import IntensityTable
from starfish.types import Features, Indices, Number, SpotAttributes
from ._base import SpotFinderAlgorithmBase
from .detect import detect_spots


class LocalMaxPeakFinder(SpotFinderAlgorithmBase):
    def __init__(self, min_distance: int, stringency: int, min_obj_area: int,
                 max_obj_area: int, threshold: int = None, measurement_type: str = 'max',
                 min_num_spots_detected: int = 3, is_volume: bool = False,
                 verbose: bool = True) -> None:

        self.min_distance = min_distance
        self.stringency = stringency
        self.min_obj_area = min_obj_area
        self.max_obj_area = max_obj_area
        self.threshold = threshold
        self.min_num_spots_detected = min_num_spots_detected

        self.measurement_function = self._get_measurement_function(measurement_type)

        self.is_volume = is_volume
        if self.is_volume:
            raise ValueError('LocalMaxPeakFinder only works for 2D data, for 3D data, '
                             'please use TrackpyLocalMaxPeakFinder')

        self.verbose = verbose

        # these parameters are useful for debugging spot-calls
        self._thresholds = None
        self._spot_counts = None
        self._grad = None
        self._spot_props = None
        self._labels = None

    def _compute_threshold(self, img: Union[np.ndarray, xr.DataArray]) -> Number:
        thresholds, spot_counts = self._compute_num_spots_per_threshold(img)
        threshold = self._select_optimal_threshold(thresholds, spot_counts)
        return threshold

    def _compute_num_spots_per_threshold(self, img: np.ndarray) -> Tuple[List, List]:

        # thresholds to search over
        thresholds = np.linspace(img.min(), img.max(), num=100)

        # number of spots detected at each threshold
        spot_counts = []

        # where we stop our threshold search
        stop_threshold = None

        if self.verbose:
            threshold_iter = tqdm(thresholds)
            print('Determining optimal threshold ...')
        else:
            threshold_iter = thresholds

        for threshold in threshold_iter:
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
            if len(spots) <= self.min_num_spots_detected:
                stop_threshold = threshold
                msg = '.. stopping early -- number of spots ' \
                      'fell below: {}'.format(self.min_num_spots_detected)
                print(msg)
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

        self._thresholds = thresholds
        self._spot_counts = spot_counts

        return thresholds, spot_counts

    def _select_optimal_threshold(self, thresholds: List, spot_counts: List) -> Number:

        # calculate the gradient of the number of spots
        grad = np.gradient(spot_counts)
        self._grad = grad
        optimal_threshold_index = np.argmin(grad)

        # only consider thresholds > than optimal threshold
        thresholds = thresholds[optimal_threshold_index:]
        grad = grad[optimal_threshold_index:]

        # if all else fails, return 0.
        selected_thr = 0

        # TODO i don't really get what this code does
        if thresholds.shape > (1,):

            distances = []

            start_point = Point(thresholds[0], grad[0])
            end_point = Point(thresholds[-1], grad[-1])
            line = Line(start_point, end_point)

            # calculate the distance between all points and the line
            for k in range(len(thresholds)):
                p = Point(thresholds[k], grad[k])
                dst = line.distance(p)
                distances.append(dst.evalf())

            # remove the end points
            thresholds = thresholds[1:-1]
            distances = distances[1:-1]

            if distances:
                thr_idx = np.argmax(np.array(distances))

                if thr_idx + self.stringency < len(thresholds):
                    selected_thr = thresholds[thr_idx + self.stringency]
                else:
                    selected_thr = thresholds[thr_idx]

        return selected_thr

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:

        if self.threshold is None:
            self.threshold = self._compute_threshold(data_image)

        # TODO @ajc data_image is volumetric, although in his code, we never use it that way
        masked_image = data_image[:, :] > self.threshold
        labels = label(masked_image)[0]
        spot_props = regionprops(labels)

        for spot_prop in spot_props:
            if spot_prop.area < self.min_obj_area or spot_prop.area > self.max_obj_area:
                masked_image[spot_prop.coords[:, 0], spot_prop.coords[:, 1]] = 0

        labels = label(masked_image)[0]
        spot_props = regionprops(labels)

        self._spot_props = spot_props
        self._labels = labels

        if self.verbose:
            print('computing final spots ...')

        spot_coords = peak_local_max(data_image,
                                     min_distance=self.min_distance,
                                     threshold_abs=self.threshold,
                                     exclude_border=False,
                                     indices=True,
                                     num_peaks=np.inf,
                                     footprint=None,
                                     labels=labels)

        self._spot_coords = spot_coords

        # TODO how to get the radius? unlikely that this can be pulled out of
        # self._spot_props, since the last call to peak_local_max can find multiple
        # peaks per label
        res = {Indices.X.value: spot_coords[:, 1],
               Indices.Y.value: spot_coords[:, 0],
               Indices.Z.value: np.zeros(len(spot_coords)),
               Features.SPOT_RADIUS: 1,
               Features.SPOT_ID: np.arange(spot_coords.shape[0]),
               Features.INTENSITY: data_image[spot_coords[:, 0], spot_coords[:, 1]]
               }

        return SpotAttributes(pd.DataFrame(res))

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
            radius_is_gyration=False,
            is_volume=self.is_volume
        )

        return intensity_table

    @classmethod
    def _add_arguments(cls, group_parser):
        group_parser.add_argument(
            "--min-distance", default=4, type=int,
            help="Minimum spot size (in number of pixels deviation)")
        group_parser.add_argument(
            "--min-obj-area", default=6, type=int,
            help="Maximum spot size (in number of pixels")
        group_parser.add_argument(
            "--max_obj_area", default=300, type=int,
            help="Maximum spot size (in number of pixels)")
        group_parser.add_argument(
            "--stringency", default=0, type=int,
            help="Number of indices in threshold list to look past "
                 "for the threhsold finding algorithm")
        group_parser.add_argument("--threshold", default=None, type=float,
                                  help="Threshold on which to threshold "
                                       "image prior to spot detection")
        group_parser.add_argument(
            "--min-num-spots-detected", default=3, type=int,
            help="Minimum number of spots detected at which to stop a"
                 "utomatic thresholding algorithm")
        group_parser.add_argument(
            "--measurement-type", default='max', type=str,
            help="How to aggregate pixel intensities in a spot")
        group_parser.add_argument(
            "--is-volume", default=False, action='store_false', help="Find spots in 3D or not")
        group_parser.add_argument(
            "--verbose", default=True, action='store_true', help="Verbosity flag")
