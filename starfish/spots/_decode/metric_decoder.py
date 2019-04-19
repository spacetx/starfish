from starfish.codebook.codebook import Codebook
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.util import click
from starfish.types import Number
from ._base import DecodeAlgorithmBase


class MetricDistance(DecodeAlgorithmBase):

    def __init__(
        self,
        codebook: Codebook,
        max_distance: Number,
        min_intensity: Number,
        norm_order: int,
        metric: str = "euclidean",
    ):
        

    def run(
        self,
        intensities: IntensityTable,
    ) -> IntensityTable:
        """Decode spots by selecting the max-valued channel in each sequencing round

        Parameters
        ----------
        intensities : IntensityTable
            IntensityTable to be decoded
        codebook : Codebook
            Contains codes to decode IntensityTable

        Returns
        -------
        IntensityTable :
            IntensityTable decoded and appended with Features.TARGET and Features.QUALITY values.

        """
        return self.codebook.decode_metric(
            intensities,
            max_distance=self.max_distance,
            min_intensity=self.min_intensity,
            norm_order=self.norm_order,
            metric=self.metric,
        )

    @staticmethod
    @click.command("MetricDistance")
    @click.pass_context
    def _cli(ctx):
        ctx.obj["component"]._cli_run(ctx, MetricDistance())
