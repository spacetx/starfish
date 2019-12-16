"""Algorithms in this module filter a BinaryMaskCollection, producing another
BinaryMaskCollection."""
from ._base import FilterAlgorithm
from .areafilter import AreaFilter
from .map import Map
from .min_distance_label import MinDistanceLabel
from .reduce import Reduce

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
__all__ = list(set(
    implementation_name
    for implementation_name, implementation_cls in locals().items()
    if isinstance(implementation_cls, type) and issubclass(implementation_cls, FilterAlgorithm)
))
