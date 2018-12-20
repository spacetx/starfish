from starfish.codebook.codebook import Codebook
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.util import click
from ._base import DecoderAlgorithmBase


class PerRoundMaxChannelDecoder(DecoderAlgorithmBase):

    def __init__(self):
        pass

    def run(self, intensities: IntensityTable, codebook: Codebook) -> IntensityTable:
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
        return codebook.decode_per_round_max(intensities)

    @staticmethod
    @click.command("PerRoundMaxChannelDecoder")
    @click.pass_context
    def _cli(ctx):
        ctx.obj["component"]._cli_run(ctx, PerRoundMaxChannelDecoder())
