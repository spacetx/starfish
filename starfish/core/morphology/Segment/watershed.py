from typing import Mapping, Optional

import numpy as np
from skimage.segmentation import watershed

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.morphology.util import _get_axes_names
from starfish.core.types import ArrayLike, Axes, Coordinates, Number
from ._base import SegmentAlgorithm


class WatershedSegment(SegmentAlgorithm):
    """Segments an image using a watershed algorithm.  This wraps scikit-image's watershed
    algorithm.

    The image being segmented must be an ImageStack with num_rounds == 1 and num_chs == 1.

    Any parameters besides image, markers, and mask should be set in the constructor and will be
    passed to scikit-image's watershed.

    See Also
    --------
    skimage.segmentation.watershed
    """
    def __init__(self, **watershed_kwargs):
        self.watershed_kwargs = watershed_kwargs

    def run(  # type: ignore
            self,
            image: ImageStack,
            markers: Optional[BinaryMaskCollection] = None,
            mask: Optional[BinaryMaskCollection] = None,
            *args, **kwargs
    ) -> BinaryMaskCollection:
        """Runs scikit-image's watershed
        """
        if image.num_rounds != 1:
            raise ValueError(
                f"{WatershedSegment.__name__} given an image with more than one round "
                f"{image.num_rounds}")
        if image.num_chs != 1:
            raise ValueError(
                f"{WatershedSegment.__name__} given an image with more than one channel "
                f"{image.num_chs}")
        if mask is not None and len(mask) != 1:
            raise ValueError(
                f"{WatershedSegment.__name__} given a mask given a mask with more than one "
                f"channel {image.num_chs}")
        if len(args) != 0 or len(kwargs) != 0:
            raise ValueError(
                f"{WatershedSegment.__name__}'s run method should not have additional arguments.")

        image_npy = 1 - image._squeezed_numpy(Axes.ROUND, Axes.CH)
        markers_npy = np.asarray(markers.to_label_image().xarray) if markers is not None else None
        mask_npy = mask.uncropped_mask(0) if mask is not None else None

        watershed_output = watershed(
            image_npy,
            markers=markers_npy,
            mask=mask_npy,
            **self.watershed_kwargs
        )

        pixel_ticks: Mapping[Axes, ArrayLike[int]] = {
            Axes(axis): axis_data
            for axis, axis_data in image.xarray.coords.items()
            if axis in _get_axes_names(3)[0]
        }
        physical_ticks: Mapping[Coordinates, ArrayLike[Number]] = {
            Coordinates(coord): coord_data
            for coord, coord_data in image.xarray.coords.items()
            if coord in _get_axes_names(3)[1]
        }

        return BinaryMaskCollection.from_label_array_and_ticks(
            watershed_output,
            pixel_ticks,
            physical_ticks,
            image.log,  # FIXME: (ttung) this should somehow include the provenance of markers and
                        # mask.
        )
