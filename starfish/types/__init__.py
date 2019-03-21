from typing import Union

from ._constants import (
    Axes,
    Clip,
    Coordinates,
    CORE_DEPENDENCIES,
    Features,
    LOG,
    PHYSICAL_COORDINATE_DIMENSION,
    PhysicalCoordinateTypes,
    STARFISH_EXTRAS_KEY
)
from ._decoded_spots import DecodedSpots
from ._spot_attributes import SpotAttributes

Number = Union[int, float]
