from abc import abstractmethod
from typing import Callable, Optional, Sequence

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.types import Number, SpotFindingResults


class FindSpotsAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(self, image_stack: ImageStack,
            reference_image: Optional[ImageStack] = None, *args) -> SpotFindingResults:
        """Find and measure spots across rounds and channels in the provided ImageStack."""
        raise NotImplementedError()

    @staticmethod
    def _get_measurement_function(measurement_type: str) -> Callable[[Sequence], Number]:
        try:
            measurement_function = getattr(np, measurement_type)
        except AttributeError:
            raise ValueError(
                f'measurement_type must be a numpy reduce function such as "max" or "mean". '
                f'{measurement_type} not found.')
        return measurement_function
