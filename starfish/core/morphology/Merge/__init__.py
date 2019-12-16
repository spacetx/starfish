"""Algorithms in this module merge multiple BinaryMaskCollections together."""
from ._base import MergeAlgorithm
from .simple import SimpleMerge

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
__all__ = list(set(
    implementation_name
    for implementation_name, implementation_cls in locals().items()
    if isinstance(implementation_cls, type) and issubclass(implementation_cls, MergeAlgorithm)
))
