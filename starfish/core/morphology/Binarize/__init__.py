"""Algorithms in this module binarize an ImageStack into a BinaryMaskCollection."""
from ._base import BinarizeAlgorithm
from .threshold import ThresholdBinarize

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
all_filters = {
    filter_name: filter_cls
    for filter_name, filter_cls in locals().items()
    if isinstance(filter_cls, type) and issubclass(filter_cls, BinarizeAlgorithm)
}
__all__ = list(all_filters.keys())
