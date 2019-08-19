from typing import Optional

import numpy as np

from starfish.core.codebook.codebook import Codebook
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import SpotAttributes
from ._base import DecodeSpotsAlgorithmBase
from .decoding_utils import convert_spot_attributes_to_traces, measure_spot_intensities


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

    def run(self, spot_attributes: SpotAttributes, image_stack: Optional[ImageStack] = None, *args):
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
        if image_stack:
            intensities = measure_spot_intensities(data_image=image_stack,
                                                   spot_attributes=spot_attributes,
                                                   measurement_function=np.mean)
        else:
            intensities = convert_spot_attributes_to_traces(spot_attributes)

        return self.codebook.decode_per_round_max(intensities)
