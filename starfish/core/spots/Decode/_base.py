from abc import abstractmethod

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class DecodeAlgorithmBase(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, intensities: IntensityTable, *args):
        """Performs decoding on the spots found, using the codebook specified."""
        raise NotImplementedError()
