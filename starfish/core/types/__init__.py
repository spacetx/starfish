from typing import Sequence, Tuple, TypeVar, Union

import numpy as np
import xarray as xr

from ._constants import (  # noqa: F401
    Axes,
    Coordinates,
    CORE_DEPENDENCIES,
    Features,
    Levels,
    LOG,
    OverlapStrategy,
    PHYSICAL_COORDINATE_DIMENSION,
    PhysicalCoordinateTypes,
    STARFISH_EXTRAS_KEY,
    TraceBuildingStrategies,
    TransformType
)
from ._decoded_spots import DecodedSpots  # noqa: F401
from ._functionsource import FunctionSource, FunctionSourceBundle  # noqa: F401
from ._spot_attributes import SpotAttributes  # noqa: F401
from ._spot_finding_results import PerImageSliceSpotResults, SpotFindingResults  # noqa: F401

Number = Union[int, float]
CoordinateValue = Union[Number, Tuple[Number, Number]]

NPArrayLike = TypeVar("NPArrayLike", np.ndarray, xr.DataArray)

ArrayLikeTypes = TypeVar("ArrayLikeTypes", int, Number)
ArrayLike = Union[xr.DataArray, np.ndarray, Sequence[ArrayLikeTypes]]
"""ArrayLike is a parameterizable custom type that includes np.ndarrays, xr.DataArrays, and
Sequences of typed values.  Once the scipy stack supports typed arrays
(https://github.com/numpy/numpy/issues/7370), we can extend that to the array types."""
