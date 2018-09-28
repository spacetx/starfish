from typing import Tuple, Union, Dict, Optional, List

import numpy as np
import xarray as xr
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.measure._regionprops import _RegionProperties

from starfish import ImageStack, IntensityTable
from starfish.spots._detector._base import SpotFinderAlgorithmBase
from starfish.spots._detector.detect import detect_spots
from starfish.types import SpotAttributes, Features, Number


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
            self.threshold = self.calculate_threshold()
        else:
            self.threshold = threshold

        if is_volume:
            raise ValueError(
                'LocalMaxPeakFinder only works for 2D data, for 3D data, please use TrackpyLocalMaxPeakFinder')

    def calculate_threshold(self, img: np.ndarray) -> Tuple[List, List]:

        # thresholds to search over
        thresholds = np.linspace(img.min(), img.max(), num=100)

        # number of spots detected at each threshold
        spot_counts = []

        # where we stopped our threshold search
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

        return thresholds, spot_counts

    def _calculate_gradient(self, thresholds, spot_counts):
        if len(spot_counts) > 1:

            # Trim the threshold array in order to match the stopping point
            # used the [0][0] to get the first number and then take it out from list
            thresholds = thresholds[:np.where(thresholds == stop_threshold)[0][0]]

            # Calculate the gradient of the number of peaks distribution
            grad = np.gradient(spot_counts)

            # Restructure the data in order to avoid to consider the min_peak in the
            # calculations

            # Coord of the gradient min_peak
            grad_min_peak_coord = np.argmin(grad)

            # Trim the data to remove the peak.
            trimmed_thr_array = thresholds[grad_min_peak_coord:]
            trimmed_grad = grad[grad_min_peak_coord:]

            if trimmed_thr_array.shape > (1,):

                # Trim the coords array in order to maintain the same length of the 
                # tr and pk
                trimmed_peaks_coords = spot_indices[grad_min_peak_coord:]
                trimmed_total_peaks = spot_counts[grad_min_peak_coord:]

                # To determine the threshold we will determine the Thr with the biggest
                # distance to the segment that join the end points of the calculated
                # gradient

                # Distances list
                distances = []

                # Calculate the coords of the end points of the gradient
                p1 = Point(trimmed_thr_array[0], trimmed_grad[0])
                p2 = Point(trimmed_thr_array[-1], trimmed_grad[-1])

                # Create a line that join the points
                s = Line(p1, p2)
                allpoints = np.arange(0, len(trimmed_thr_array))

                # Calculate the distance between all points and the line
                for p in allpoints:
                    dst = s.distance(Point(trimmed_thr_array[p], trimmed_grad[p]))
                    distances.append(dst.evalf())

                # Remove the end points from the lists
                trimmed_thr_array = trimmed_thr_array[1:-1]
                trimmed_grad = trimmed_grad[1:-1]
                trimmed_peaks_coords = trimmed_peaks_coords[1:-1]
                trimmed_total_peaks = trimmed_total_peaks[1:-1]
                trimmed_distances = distances[1:-1]

                # Determine the coords of the selected Thr
                # Converted trimmed_distances to array because it crashed
                # on Sanger.
                if trimmed_distances:  # Most efficient way will be to consider the length of Thr list
                    thr_idx = np.argmax(np.array(trimmed_distances))
                    calculated_thr = trimmed_thr_array[thr_idx]
                    # The selected threshold usually causes oversampling of the number of dots
                    # I added a stringency parameter (int n) to use to select the Thr+n 
                    # for the counting. It selects a stringency only if the trimmed_thr_array
                    # is long enough
                    if thr_idx + stringency < len(trimmed_thr_array):
                        selected_thr = trimmed_thr_array[thr_idx + stringency]
                        selected_peaks = trimmed_peaks_coords[thr_idx + stringency]
                        thr_idx = thr_idx + stringency
                    else:
                        selected_thr = trimmed_thr_array[thr_idx]
                        selected_peaks = trimmed_peaks_coords[thr_idx]
        else:
            selected_thr = 0

        return selected_thr


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

    @staticmethod
    def _single_spot_attributes(
            spot_property: _RegionProperties,
            decoded_image: np.ndarray,
            min_area: Number,
            max_area: Number,
    ) -> Tuple[Dict[str, int], int]:
        """
        Calculate starfish SpotAttributes from the RegionProperties of a connected component
        feature.

        Parameters
        ----------
        spot_property: _RegionProperties
            Properties of the connected component. Output of skimage.measure.regionprops
        decoded_image : np.ndarray
            Image whose pixels correspond to the targets that the given position in the ImageStack
            decodes to.
        target_map : TargetsMap
            Unique mapping between string target names and int target IDs.
        min_area :
            Combined features with area below this value are marked as failing filters
        max_area : Number
            Combined features with area above this value are marked as failing filters

        Returns
        -------
        Dict[str, Number] :
            spot attribute dictionary for this connected component, containing the x, y, z position,
            target name (str) and feature radius.
        int :
            1 if spot passes size filters, zero otherwise.

        """
        # because of the above skimage issue, we need to support both 2d and 3d properties
        if len(spot_property.centroid) == 3:
            spot_attrs = {
                'z': int(spot_property.centroid[0]),
                'y': int(spot_property.centroid[1]),
                'x': int(spot_property.centroid[2])
            }
        else:  # data is 2d
            spot_attrs = {
                'z': 0,
                'y': int(spot_property.centroid[0]),
                'x': int(spot_property.centroid[1])
            }

        # we're back to 3d or fake-3d here
        target_index = decoded_image[spot_attrs['z'], spot_attrs['y'], spot_attrs['x']]
        spot_attrs[Features.TARGET] = target_map.target_as_str(target_index)
        spot_attrs[Features.SPOT_RADIUS] = spot_property.equivalent_diameter / 2

        # filter intensities for which radius is too small
        passes_area_filter = 1 if min_area <= spot_property.area < max_area else 0
        return spot_attrs, passes_area_filter

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
