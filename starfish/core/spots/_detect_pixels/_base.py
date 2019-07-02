from abc import abstractmethod
from typing import Callable, Sequence, Tuple, Type

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.pipeline.pipelinecomponent import PipelineComponent
from starfish.core.types import Number
from .combine_adjacent_features import ConnectedComponentDecodingResult


class DetectPixels(PipelineComponent):
    """
    Decode an image by first coding each pixel, then combining the results into spots.
    """
    pass


class DetectPixelsAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return DetectPixels

    @abstractmethod
    def run(
            self,
            primary_image: ImageStack,
            *args,
    ) -> Tuple[IntensityTable, ConnectedComponentDecodingResult]:
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
