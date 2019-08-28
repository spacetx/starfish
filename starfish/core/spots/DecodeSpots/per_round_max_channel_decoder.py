from typing import Optional

import numpy as np

from starfish.core.codebook.codebook import Codebook
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.spots.DecodeSpots.decoding_uitls import build_spot_traces_exact_match
from starfish.core.types import SpotAttributes
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

    def run(self, spot_attributes: SpotAttributes, image_stack, *args) -> DecodedIntensityTable:
        """Decode spots by selecting the max-valued channel in each sequencing round

        Parameters
        ----------
        spot_attributes

        image_stack

        Returns
        -------
        IntensityTable :
            IntensityTable decoded and appended with Features.TARGET and Features.QUALITY values.

        """
        intensities = build_spot_traces_exact_match(spot_attributes, imagestack=image_stack)
        return self.codebook.decode_per_round_max(intensities)
