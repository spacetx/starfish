from typing import Callable, Optional, Sequence,  Union

import numpy as np
import xarray as xr

from starfish.intensity_table import IntensityTable
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.stack import ImageStack
from starfish.types import Number, SpotAttributes


class SpotFinderAlgorithmBase(AlgorithmBase):
    def run (
            self, stack: ImageStack,
            blobs_image: Optional[Union[np.ndarray, xr.DataArray]] = None,
            reference_image_from_max_projection: bool = False) \
            -> IntensityTable:
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
