from typing import Tuple, Union, Number, Dict, Optional

import numpy as np
import xarray as xr
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.measure._regionprops import _RegionProperties

from starfish import ImageStack, IntensityTable
from starfish.spots._detector._base import SpotFinderAlgorithmBase
from starfish.spots._detector.detect import detect_spots
from starfish.types import SpotAttributes


class LocalMaxPeakFinder(SpotFinderAlgorithmBase):
    def __init__(
            self, min_distance, stringency, min_obj_area, max_obj_area, threshold=None
            , measurement_type: str = 'max', is_volume: bool = False,
            verbose=False, **kwargs) -> None:

        self.min_distance = min_distance
        self.stringency = stringency
        self.min_obj_area = min_obj_area
        self.max_obj_area = max_obj_area

        self.threshold = self = threshold

        if (is_volume):
            raise ValueError(
                'LocalMaxPeakFinder only works for 2D data, for 3D data, please use TrackpyLocalMaxPeakFinder')

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
            blobs_image: Optional[Union[np.ndarray, xr.DataArray]]=None,
            reference_image_from_max_projection: bool=False,
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