from typing import Callable

from starfish.core.codebook.codebook import Codebook
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_to_intensity_table
from starfish.core.spots.DecodeSpots.trace_builders import trace_builders
from starfish.core.types import SpotFindingResults, TraceBuildingStrategies
from ._base import DecodeSpotsAlgorithm


class PerRoundMaxChannel(DecodeSpotsAlgorithm):
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

    def __init__(
            self,
            codebook: Codebook,
            anchor_round: int=1,
            search_radius: int=3,
            trace_building_strategy: TraceBuildingStrategies=TraceBuildingStrategies.EXACT_MATCH):
        self.codebook = codebook
        self.trace_builder: Callable = trace_builders[trace_building_strategy]
        self.anchor_round = anchor_round
        self.search_radius = search_radius

    def run(self, spots: SpotFindingResults, *args) -> DecodedIntensityTable:
        """Decode spots by selecting the max-valued channel in each sequencing round

        Parameters
        ----------
        spots: SpotFindingResults
            A Dict of tile indices and their corresponding measured spots

        Returns
        -------
        DecodedIntensityTable :
            IntensityTable decoded and appended with Features.TARGET and Features.QUALITY values.

        """
        intensities = self.trace_builder(spot_results=spots,
                                         anchor_round=self.anchor_round,
                                         search_radius=self.search_radius)
        transfer_physical_coords_to_intensity_table(intensity_table=intensities, spots=spots)
        return self.codebook.decode_per_round_max(intensities)
