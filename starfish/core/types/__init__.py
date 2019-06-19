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
    TransformType
)
from ._decoded_spots import DecodedSpots
from ._spot_attributes import SpotAttributes

Number = Union[int, float]
CoordinateValue = Union[Number, Tuple[Number, Number]]
