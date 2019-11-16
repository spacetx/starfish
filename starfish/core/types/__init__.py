from typing import Sequence, Tuple, TypeVar, Union

import numpy as np
import xarray as xr

from ._constants import (
    Axes,
    Clip,
    Coordinates,
    CORE_DEPENDENCIES,
    Features,
    LOG,
    OverlapStrategy,
    PHYSICAL_COORDINATE_DIMENSION,
    PhysicalCoordinateTypes,
    STARFISH_EXTRAS_KEY,
    TraceBuildingStrategies,
    TransformType
)
from ._decoded_spots import DecodedSpots
from ._functionsource import FunctionSource
from ._spot_attributes import SpotAttributes
from ._spot_finding_results import PerImageSliceSpotResults, SpotFindingResults

Number = Union[int, float]
CoordinateValue = Union[Number, Tuple[Number, Number]]

ArrayLikeTypes = TypeVar("ArrayLikeTypes", int, Number)
ArrayLike = Union[np.ndarray, xr.DataArray, Sequence[ArrayLikeTypes]]
"""ArrayLike is a parameterizable custom type that includes np.ndarrays, xr.DataArrays, and
Sequences of typed values.  Once the scipy stack supports typed arrays
(https://github.com/numpy/numpy/issues/7370), we can extend that to the array types."""
