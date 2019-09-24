from typing import Tuple, Union

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
from ._spot_finding_results import SpotFindingResults

Number = Union[int, float]
CoordinateValue = Union[Number, Tuple[Number, Number]]
