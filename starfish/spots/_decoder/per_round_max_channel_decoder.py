from starfish.codebook.codebook import Codebook
from starfish.intensity_table.intensity_table import IntensityTable
from ._base import DecoderAlgorithmBase


class PerRoundMaxChannelDecoder(DecoderAlgorithmBase):

    def __init__(self, **kwargs):
        pass

    @classmethod
    def _add_arguments(cls, group_parser):
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
