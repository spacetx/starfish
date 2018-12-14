import collections
from typing import Mapping

from starfish.types import Indices

_DimensionMetadata = collections.namedtuple("_DimensionMetadata", ['order', 'required'])

AXES_DATA: Mapping[Indices, _DimensionMetadata] = {
    Indices.ROUND: _DimensionMetadata(0, True),
    Indices.CH: _DimensionMetadata(1, True),
    Indices.Z: _DimensionMetadata(2, False),
}
N_AXES = max(data.order for data in AXES_DATA.values()) + 1
