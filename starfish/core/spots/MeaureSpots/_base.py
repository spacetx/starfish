from abc import abstractmethod

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.types import SpotAttributes


class MeasureSpotsAlgorithmBase(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, image_stack: ImageStack, spot_locations: SpotAttributes, *args
            ) -> IntensityTable:
        """Measures the intensity of spots and given x/y/z locations."""
        raise NotImplementedError()
