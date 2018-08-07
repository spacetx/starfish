from typing import Callable, Sequence, Union

import numpy as np
import xarray as xr

from starfish.image import ImageStack
from starfish.intensity_table import IntensityTable
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.features.spot_attributes import SpotAttributes
from starfish.types import Number


class SpotFinderAlgorithmBase(AlgorithmBase):
    def find(
            self,
            hybridization_image: ImageStack,
            blobs_image: Union[np.ndarray, xr.DataArray]
    ) -> IntensityTable:
        """Finds spots in an ImageStack"""
        raise NotImplementedError()

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:
        """Finds spots in a 3d volume"""
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
