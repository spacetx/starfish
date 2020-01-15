"""Algorithms in this module segment cells and store the results in a BinaryMaskCollection."""
from ._base import SegmentAlgorithm
from .watershed import WatershedSegment

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
__all__ = list(set(
    implementation_name
    for implementation_name, implementation_cls in locals().items()
    if isinstance(implementation_cls, type) and issubclass(implementation_cls, SegmentAlgorithm)
))
