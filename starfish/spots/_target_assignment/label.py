import numpy as np

from starfish.intensity_table.intensity_table import IntensityTable
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
        label_image: np.ndarray,
        intensities: IntensityTable,
        in_place: bool,
    ) -> IntensityTable:

        if len(label_image.shape) == 3:
            cell_ids = label_image[
                intensities[Axes.ZPLANE.value].values,
                intensities[Axes.Y.value].values,
                intensities[Axes.X.value].values
            ]
        elif len(label_image.shape) == 2:
            cell_ids = label_image[
                intensities[Axes.Y.value].values,
                intensities[Axes.X.value].values
            ]
        else:
            raise ValueError(
                f"`label_image` must be 2 or 3 dimensional, not {len(label_image.shape)}D."
            )

        if not in_place:
            intensities = intensities.copy()

        intensities[Features.CELL_ID] = (Features.AXIS, cell_ids)

        return intensities

    def run(
            self, label_image: np.ndarray, intensity_table: IntensityTable, verbose: bool=False,
            in_place: bool=False,
    ) -> IntensityTable:
        """Extract cell ids for features in IntensityTable from a segmentation label image

        Parameters
        ----------
        label_image : np.ndarray[np.uint32]
            integer array produced from segmentation where each pixel in a cell is labeled by the
            same integer, and each cell is labeled by a different integer
        intensities : IntensityTable
            spot information

        Returns
        -------
        IntensityTable :
            IntensityTable with added features variable containing cell ids. Points outside of
            cells will be assigned zero.

        """
        return self._assign(label_image, intensity_table, in_place=in_place)

    @staticmethod
    @click.command("Label")
    @click.pass_context
    def _cli(ctx):
        ctx.obj["component"]._cli_run(ctx, Label())
