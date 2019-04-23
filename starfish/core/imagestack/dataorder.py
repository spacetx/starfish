import collections
from typing import Mapping

from starfish.core.types import Axes

_DimensionMetadata = collections.namedtuple("_DimensionMetadata", ['order', 'required'])

AXES_DATA: Mapping[Axes, _DimensionMetadata] = {
    Axes.ROUND: _DimensionMetadata(0, True),
    Axes.CH: _DimensionMetadata(1, True),
    Axes.ZPLANE: _DimensionMetadata(2, False),
}
N_AXES = max(data.order for data in AXES_DATA.values()) + 1
