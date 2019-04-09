from starfish.intensity_table.intensity_table import IntensityTable
from starfish.segmentation_mask import SegmentationMaskCollection
from starfish.types import Axes, Features
from starfish.util import click
from ._base import TargetAssignmentAlgorithm


class Label(TargetAssignmentAlgorithm):

    def __init__(self, **kwargs) -> None:
        """
        Label accepts no parameters, but all pipeline components must accept arbitrary kwargs
        """

    @classmethod
    def _add_arguments(cls, parser) -> None:
        pass

    @staticmethod
    def _assign(
        masks: SegmentationMaskCollection,
        intensities: IntensityTable,
        in_place: bool,
    ) -> IntensityTable:
        cell_ids = []

        # for each spot, test whether the spot falls inside the area of each mask
        for spot in intensities:
            for mask in masks:
                sel = {Axes.X.value: spot[Axes.X.value],
                       Axes.Y.value: spot[Axes.Y.value]}
                if mask.ndim == 3:
                    sel[Axes.ZPLANE.value] = spot[Axes.ZPLANE.value]

                try:
                    if mask.sel(sel):
                        cell_id = mask.name
                        break
                except KeyError:
                    pass
            else:
                cell_id = ''

            cell_ids.append(cell_id)

        if not in_place:
            intensities = intensities.copy()

        intensities[Features.CELL_ID] = (Features.AXIS, cell_ids)

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
        masks : SegmentaionMaskCollection
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
