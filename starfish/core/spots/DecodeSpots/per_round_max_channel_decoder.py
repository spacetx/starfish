from typing import Callable, Optional

from starfish.core.codebook.codebook import Codebook
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_to_intensity_table
from starfish.core.spots.DecodeSpots.trace_builders import TRACE_BUILDERS
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
    trace_building_strategy: TraceBuildingStrategies
        Defines the strategy for building spot traces to decode across rounds and chs of spot
        finding results.
    search_radius : Optional[int]
        Only applicable if trace_building_strategy is TraceBuildingStrategies.NEAREST_NEIGHBORS.
        Number of pixels over which to search for spots in other rounds and channels.
    anchor_round : Optional[int]
        Only applicable if trace_building_strategy is TraceBuildingStrategies.NEAREST_NEIGHBORS.
        The imaging round against which other rounds will be checked for spots in the same
        approximate pixel location.
    """

    def __init__(
            self,
            codebook: Codebook,
            trace_building_strategy: TraceBuildingStrategies = TraceBuildingStrategies.EXACT_MATCH,
            anchor_round: Optional[int]=1,
            search_radius: Optional[int]=3):
        self.codebook = codebook
        self.trace_builder: Callable = TRACE_BUILDERS[trace_building_strategy]
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
