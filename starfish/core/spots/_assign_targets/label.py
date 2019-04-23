import numpy as np

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.segmentation_mask import SegmentationMaskCollection
from starfish.core.types import Features
from starfish.core.util import click
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
        intensities: IntensityTable,
        in_place: bool,
    ) -> IntensityTable:

        intensities[Features.CELL_ID] = (
            Features.AXIS,
            np.full(intensities.sizes[Features.AXIS], fill_value='nan', dtype='<U8')
        )

        for mask in masks:
            y_min, y_max = float(mask.y.min()), float(mask.y.max())
            x_min, x_max = float(mask.x.min()), float(mask.x.max())

            in_bbox = intensities.where(
                (intensities.y >= y_min)
                & (intensities.y <= y_max)
                & (intensities.x >= x_min)
                & (intensities.x <= x_max),
                drop=True
            )

            in_mask = mask.sel_points(y=in_bbox.y, x=in_bbox.x)
            spot_ids = in_bbox[Features.SPOT_ID][in_mask.values]
            intensities[Features.CELL_ID].loc[spot_ids] = mask.name

        return intensities

    def run(
            self,
            masks: SegmentationMaskCollection,
            intensity_table: IntensityTable,
            verbose: bool = False,
            in_place: bool = False,
    ) -> IntensityTable:
        """Extract cell ids for features in IntensityTable from a segmentation label image

        Parameters
        ----------
        masks : SegmentationMaskCollection
            binary masks segmenting each cell
        intensity_table : IntensityTable
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
        return self._assign(masks, intensity_table, in_place=in_place)

    @staticmethod
    @click.command("Label")
    @click.pass_context
    def _cli(ctx):
        ctx.obj["component"]._cli_run(ctx, Label())
