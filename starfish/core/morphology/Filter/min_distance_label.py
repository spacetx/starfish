import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from ._base import FilterAlgorithm


class MinDistanceLabel(FilterAlgorithm):
    """
    Using a BinaryMaskCollection with a single mask, produce a new BinaryMaskCollection with a mask
    representing each feature.  Features are determined by obtaining the peak local maximum of a
    matrix consisting of the distance from each position to background.  Watershed is applied to the
    distance matrix to obtain the features.

    Note that due to bugs in skimage, exclude_border should generally be set to False.

    Parameters
    ----------
    minimum_distance_xy : int
        The minimum distance between the peaks along the x or y axis. (default: 1)
    minimum_distance_z : int
        The minimum distance between the peaks along the z axis. (default: 1)
    exclude_border : bool
        Exclude the borders for consideration for peaks. (default: False)
    """

    def __init__(
            self,
            minimum_distance_xy: int = 1,
            minimum_distance_z: int = 1,
            exclude_border: bool = False,
    ) -> None:
        self._minimum_distance_xy = minimum_distance_xy
        self._minimum_distance_z = minimum_distance_z
        self._exclude_border = exclude_border

    def run(
            self,
            binary_mask_collection: BinaryMaskCollection,
            *args,
            **kwargs
    ) -> BinaryMaskCollection:
        assert len(binary_mask_collection) == 1
        mask = binary_mask_collection.uncropped_mask(0)

        # calculates the distance of every pixel to the nearest background (0) point
        distance: np.ndarray = distance_transform_edt(mask)

        footprint = np.ones(
            shape=(
                self._minimum_distance_z * 2 + 1,
                self._minimum_distance_xy * 2 + 1,
                self._minimum_distance_xy * 2 + 1,
            ),
            dtype=bool,
        )

        # boolean array marking local maxima, excluding any maxima within min_dist
        local_maximum: np.ndarray = peak_local_max(
            distance,
            exclude_border=self._exclude_border,
            footprint=footprint,
            labels=np.asarray(mask),
        )
        local_maximum_mask = np.zeros_like(distance, dtype=bool)
        local_maximum_mask[tuple(local_maximum.T)] = True

        # label the maxima for watershed
        markers, _ = label(local_maximum_mask)

        # run watershed, using the distances in the thresholded image as basins.
        # Uses the original image as a mask, preventing any background pixels from being labeled
        labeled_array: np.ndarray = watershed(-distance, markers, mask=mask)

        return BinaryMaskCollection.from_label_array_and_ticks(
            labeled_array,
            binary_mask_collection._pixel_ticks,
            binary_mask_collection._physical_ticks,
            binary_mask_collection.log,
        )
