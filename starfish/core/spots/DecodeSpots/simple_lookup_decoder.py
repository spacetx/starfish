from typing import Dict, Tuple

from starfish.core.codebook.codebook import Codebook
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.spots.DecodeSpots.trace_builders import build_traces_sequential
from starfish.core.types import Axes, Features, SpotFindingResults
from ._base import DecodeSpotsAlgorithm


class SimpleLookupDecoder(DecodeSpotsAlgorithm):
    """
    Decode spots by assigning the target value of a spot to the corresponding target value of the
    round/ch it was found in. This method only makes sense to use in non mulitplexed sequential
    assays where each r/ch pair only has one target assigned to it.

    Parameters
    ----------
    codebook : Codebook
        Contains codes to decode IntensityTable

    """

    def __init__(self, codebook: Codebook):
        self.codebook = codebook

    def run(self, spots: SpotFindingResults, *args) -> DecodedIntensityTable:
        """
        Decode spots by looking up the associated target value for the round and ch each spot is
        in.

        Parameters
        ----------
        spots: SpotFindingResults
            A Dict of tile indices and their corresponding measured spots

        Returns
        -------
        DecodedIntensityTable :
            IntensityTable decoded and appended with Features.TARGET and values.

        """
        lookup_table: Dict[Tuple, str] = {}
        for target in self.codebook[Features.TARGET]:
            for ch_label in self.codebook[Axes.CH.value]:
                for round_label in self.codebook[Axes.ROUND.value]:
                    if self.codebook.loc[target, round_label, ch_label]:
                        lookup_table[(int(round_label), int(ch_label))] = str(target.values)

        for r_ch_index, results in spots.items():
            target = lookup_table[r_ch_index] if r_ch_index in lookup_table else 'nan'
            results.spot_attrs.data[Features.TARGET] = target
        intensities = build_traces_sequential(spots)
        return DecodedIntensityTable(intensities)
