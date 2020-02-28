from abc import abstractmethod
from typing import Callable, Sequence, Tuple

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.types import Number
from .combine_adjacent_features import ConnectedComponentDecodingResult


class DetectPixelsAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(
            self,
            primary_image: ImageStack,
            *args,
    ) -> Tuple[DecodedIntensityTable, ConnectedComponentDecodingResult]:
        """Finds spots in an ImageStack"""
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
