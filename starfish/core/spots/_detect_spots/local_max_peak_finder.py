from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from sympy import Line, Point
from tqdm import tqdm

from starfish.core.config import StarfishConfig
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, Number, SpotAttributes
from starfish.core.util import click
from ._base import DetectSpotsAlgorithmBase
from .detect import detect_spots


# TODO ambrosejcarr, ttung: https://github.com/spacetx/starfish/issues/708
# Currently, spot finders cannot propagate state, which makes the flow for this
# spot finder confusing. One would expect to have access to the private parameters
# however, they are lost due to the memory-space forking induced by multi-processing.

class LocalMaxPeakFinder(DetectSpotsAlgorithmBase):
    """
    2-dimensional local-max peak finder that wraps skimage.feature.peak_local_max

    Parameters
    ----------
    min_distance : int
        Minimum number of pixels separating peaks in a region of 2 * min_distance + 1
        (i.e. peaks are separated by at least min_distance). To find the maximum number of
        peaks, use min_distance=1.
    stringency : int
    min_obj_area : int
    max_obj_area : int
    threshold : Optional[Number]
    measurement_type : str, {'max', 'mean'}
        default 'max' calculates the maximum intensity inside the object
    min_num_spots_detected : int
        When fewer than this number of spots are detected, spot searching for higher threshold
        values. (default = 3)
    is_volume : bool
        Not supported. For 3d peak detection please use TrackpyLocalMaxPeakFinder.
        (default=False)
    verbose : bool
        If True, report the percentage completed (default = False) during processing

    Notes
    -----
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max
    """

    def __init__(
        self, min_distance: int, stringency: int, min_obj_area: int, max_obj_area: int,
        threshold: Optional[Number]=None, measurement_type: str='max',
        min_num_spots_detected: int=3, is_volume: bool=False, verbose: bool=True
    ) -> None:

        self.min_distance = min_distance
        self.stringency = stringency
        self.min_obj_area = min_obj_area
        self.max_obj_area = max_obj_area
        self.threshold = threshold
        self.min_num_spots_detected = min_num_spots_detected

        self.measurement_function = self._get_measurement_function(measurement_type)

        self.is_volume = is_volume
        self.verbose = verbose

        # these parameters are useful for debugging spot-calls
        self._thresholds: Optional[np.ndarray] = None
        self._spot_counts: Optional[List[int]] = None
        self._grad = None
        self._spot_props = None
        self._labels = None

    def _compute_num_spots_per_threshold(self, img: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Computes the number of detected spots for each threshold

        Parameters
        ----------
        img : np.ndarray
            The image in which to count spots

        Returns
        -------
        np.ndarray :
            thresholds
        List[int] :
            spot counts
        """

        # thresholds to search over
        thresholds = np.linspace(img.min(), img.max(), num=100)

        # number of spots detected at each threshold
        spot_counts = []

        # where we stop our threshold search
        stop_threshold = None

        if self.verbose and StarfishConfig().verbose:
            threshold_iter = tqdm(thresholds)
            print('Determining optimal threshold ...')
        else:
            threshold_iter = thresholds

        for stop_index, threshold in enumerate(threshold_iter):
            spots = peak_local_max(
                img,
                min_distance=self.min_distance,
                threshold_abs=threshold,
                exclude_border=False,
                indices=True,
                num_peaks=np.inf,
                footprint=None,
                labels=None
            )

            # stop spot finding when the number of detected spots falls below min_num_spots_detected
            if len(spots) <= self.min_num_spots_detected:
                stop_threshold = threshold
                if self.verbose:
                    print(f'Stopping early at threshold={threshold}. Number of spots fell below: '
                          f'{self.min_num_spots_detected}')
                break
            else:
                spot_counts.append(len(spots))

        if stop_threshold is None:
            stop_threshold = thresholds.max()

        if len(thresholds > 1):
            thresholds = thresholds[:stop_index]
            spot_counts = spot_counts[:stop_index]

        # TODO ambrosejcarr: these two lists should be a dict
        self._thresholds = thresholds
        self._spot_counts = spot_counts

        return thresholds, spot_counts

    def _select_optimal_threshold(self, thresholds: np.ndarray, spot_counts: List[int]) -> float:
        # calculate the gradient of the number of spots
        grad = np.gradient(spot_counts)
        self._grad = grad
        optimal_threshold_index = np.argmin(grad)

        # only consider thresholds > than optimal threshold
        thresholds = thresholds[optimal_threshold_index:]
        grad = grad[optimal_threshold_index:]

        # if all else fails, return 0.
        selected_thr = 0

        if len(thresholds) > 1:

            distances = []

            # create a line whose end points are the threshold and and corresponding gradient value
            # for spot_counts corresponding to the threshold
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

            # select the threshold that has the maximum distance from the line
            # if stringency is passed, select a threshold that is n steps higher, where n is the
            # value of stringency
            if distances:
                thr_idx = np.argmax(np.array(distances))

                if thr_idx + self.stringency < len(thresholds):
                    selected_thr = thresholds[thr_idx + self.stringency]
                else:
                    selected_thr = thresholds[thr_idx]

        return selected_thr

    def _compute_threshold(self, img: Union[np.ndarray, xr.DataArray]) -> float:
        """Finds spots on a number of thresholds then selects and returns the optimal threshold

        Parameters
        ----------
        img: Union[np.ndarray, xr.DataArray]
            data array in which spots should be detected and over which to compute different
            intensity thresholds

        Returns
        -------
        Number :  #TODO ambrosejcarr this should probably be a float
            The intensity threshold
        """
        img = np.asarray(img)
        thresholds, spot_counts = self._compute_num_spots_per_threshold(img)
        threshold = self._select_optimal_threshold(thresholds, spot_counts)
        return threshold

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:
        """measure attributes of spots detected by binarizing the image using the selected threshold

        Parameters
        ----------
        data_image : Union[np.ndarray, xr.DataArray]
            image containing spots to be detected

        Returns
        -------
        SpotAttributes
            Attributes for each detected spot
        """

        if self.threshold is None:
            self.threshold = self._compute_threshold(data_image)

        data_image = np.asarray(data_image)

        # identify each spot's size by binarizing and calculating regionprops
        masked_image = data_image[:, :] > self.threshold
        labels = label(masked_image)[0]
        spot_props = regionprops(np.squeeze(labels))

        # mask spots whose areas are too small or too large
        for spot_prop in spot_props:
            if spot_prop.area < self.min_obj_area or spot_prop.area > self.max_obj_area:
                masked_image[0, spot_prop.coords[:, 0], spot_prop.coords[:, 1]] = 0

        # store re-calculated regionprops and labels based on the area-masked image
        self._labels = label(masked_image)[0]
        self._spot_props = regionprops(np.squeeze(self._labels))

        if self.verbose:
            print('computing final spots ...')

        self._spot_coords = peak_local_max(
            data_image,
            min_distance=self.min_distance,
            threshold_abs=self.threshold,
            exclude_border=False,
            indices=True,
            num_peaks=np.inf,
            footprint=None,
            labels=self._labels
        )

        # TODO how to get the radius? unlikely that this can be pulled out of
        # self._spot_props, since the last call to peak_local_max can find multiple
        # peaks per label
        res = {Axes.X.value: self._spot_coords[:, 2],
               Axes.Y.value: self._spot_coords[:, 1],
               Axes.ZPLANE.value: self._spot_coords[:, 0],
               Features.SPOT_RADIUS: 1,
               Features.SPOT_ID: np.arange(self._spot_coords.shape[0]),
               Features.INTENSITY: data_image[self._spot_coords[:, 0],
                                              self._spot_coords[:, 1],
                                              self._spot_coords[:, 2]]
               }

        return SpotAttributes(pd.DataFrame(res))

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
    @click.command("LocalMaxPeakFinder")
    @click.option(
        "--min-distance", default=4, type=int,
        help="Minimum spot size (in number of pixels deviation)")
    @click.option(
        "--min-obj-area", default=6, type=int,
        help="Maximum spot size (in number of pixels")
    @click.option(
        "--max_obj_area", default=300, type=int,
        help="Maximum spot size (in number of pixels)")
    @click.option(
        "--stringency", default=0, type=int,
        help="Number of indices in threshold list to look past "
             "for the threhsold finding algorithm")
    @click.option(
        "--threshold", default=None, type=float,
        help="Threshold on which to threshold "
             "image prior to spot detection")
    @click.option(
        "--min-num-spots-detected", default=3, type=int,
        help="Minimum number of spots detected at which to stop a"
             "utomatic thresholding algorithm")
    @click.option(
        "--measurement-type", default='max', type=str,
        help="How to aggregate pixel intensities in a spot")
    @click.option(
        "--is-volume", default=False, help="Find spots in 3D or not")
    @click.option(
        "--verbose", default=True, help="Verbosity flag")
    @click.pass_context
    def _cli(ctx, min_distance, min_obj_area, max_obj_area, stringency, threshold,
             min_num_spots_detected, measurement_type, is_volume, verbose):
        instance = LocalMaxPeakFinder(min_distance, min_obj_area, max_obj_area,
                                      stringency, threshold,
                                      min_num_spots_detected, measurement_type, is_volume, verbose)
        ctx.obj["component"]._cli_run(ctx, instance)
