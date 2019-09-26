from abc import abstractmethod

from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.types import SpotFindingResults


class DecodeSpotsAlgorithm(metaclass=AlgorithmBase):
    """Performs decoding on the spots found, using the codebook specified."""

    @abstractmethod
    def run(self, spot_attributes: SpotFindingResults, *args) -> DecodedIntensityTable:
        raise NotImplementedError()
