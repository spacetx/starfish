import numpy as np

from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.segmentation_mask import SegmentationMaskCollection
from starfish.core.types import Features
from ._base import AssignTargetsAlgorithm


class Label(AssignTargetsAlgorithm):
    """
    Extract cell ids for features in IntensityTable from a set of segmentation masks.
    """

    def __init__(self, **kwargs) -> None:
        pass

    @classmethod
    def _add_arguments(cls, parser) -> None:
        pass

    @staticmethod
    def _assign(
        masks: SegmentationMaskCollection,
        decoded_intensities: DecodedIntensityTable,
        in_place: bool,
    ) -> DecodedIntensityTable:

        cell_ids = (
            Features.AXIS,
            np.full(decoded_intensities.sizes[Features.AXIS], fill_value='nan', dtype='<U8')
        )

        decoded_intensities[Features.CELL_ID] = cell_ids

        for mask in masks:
            y_min, y_max = float(mask.y.min()), float(mask.y.max())
            x_min, x_max = float(mask.x.min()), float(mask.x.max())

            in_bbox = decoded_intensities.where(
                (decoded_intensities.y >= y_min)
                & (decoded_intensities.y <= y_max)
                & (decoded_intensities.x >= x_min)
                & (decoded_intensities.x <= x_max),
                drop=True
            )

            in_mask = mask.sel(y=in_bbox.y, x=in_bbox.x)
            spot_ids = in_bbox[Features.SPOT_ID][in_mask.values]
            decoded_intensities[Features.CELL_ID].loc[spot_ids] = mask.name

        return decoded_intensities

    def run(
            self,
            masks: SegmentationMaskCollection,
            decoded_intensity_table: DecodedIntensityTable,
            verbose: bool = False,
            in_place: bool = False,
    ) -> DecodedIntensityTable:
        """Extract cell ids for features in IntensityTable from a segmentation label image

        Parameters
        ----------
        masks : SegmentationMaskCollection
            binary masks segmenting each cell
        decoded_intensity_table : IntensityTable
            spot information
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on the percentage completed during processing (default = False)

        Returns
        -------
        IntensityTable :
            IntensityTable with added features variable containing cell ids. Points outside of
            cells will be assigned zero.

        """
        return self._assign(masks, decoded_intensity_table, in_place=in_place)
