"""Algorithms in this module binarize an ImageStack into a BinaryMaskCollection."""
from ._base import BinarizeAlgorithm
from .threshold import ThresholdBinarize

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
__all__ = list(set(
    implementation_name
    for implementation_name, implementation_cls in locals().items()
    if isinstance(implementation_cls, type) and issubclass(implementation_cls, BinarizeAlgorithm)
))
