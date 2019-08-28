from abc import abstractmethod
from typing import Optional

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.types import SpotAttributes


class DecodeSpotsAlgorithmBase(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, spot_attributes: SpotAttributes, *args) -> DecodedIntensityTable:
        """Performs decoding on the spots found, using the codebook specified."""
        raise NotImplementedError()
