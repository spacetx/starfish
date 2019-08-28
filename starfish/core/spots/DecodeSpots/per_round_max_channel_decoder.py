from starfish.core.codebook.codebook import Codebook
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.spots.DecodeSpots.decoding_uitls import build_spot_traces_exact_match, build_spot_traces_nearest_neighbor
from starfish.core.types import SpotFindingResults
from ._base import DecodeSpotsAlgorithmBase


class PerRoundMaxChannel(DecodeSpotsAlgorithmBase):
    """
    Decode spots by selecting the max-valued channel in each sequencing round.

    Note that this assumes that the codebook contains only one "on" channel per sequencing round,
    a common pattern in experiments that assign one fluorophore to each DNA nucleotide and
    read DNA sequentially. It is also a characteristic of single-molecule FISH and RNAscope
    codebooks.

    Parameters
    ----------
    codebook : Codebook
        Contains codes to decode IntensityTable

    """

    def __init__(self, codebook: Codebook):
        self.codebook = codebook

    def run(self, spots: SpotFindingResults,
            exact_match: bool = True,
            search_radius: int = 3,
            anchor_round: int= 1,
            *args) -> DecodedIntensityTable:
        """Decode spots by selecting the max-valued channel in each sequencing round

        Parameters
        ----------
        spots


        Returns
        -------
        DecodedIntensityTable :
            IntensityTable decoded and appended with Features.TARGET and Features.QUALITY values.

        """
        if exact_match:
            intensities = build_spot_traces_exact_match(spots)
        else:
            intensities = build_spot_traces_nearest_neighbor(spot_results=spots,
                                                             search_radius=search_radius,
                                                             anchor_round=anchor_round)
        return self.codebook.decode_per_round_max(intensities)
