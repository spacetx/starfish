from ._base import FindSpotsAlgorithm
from .blob import BlobDetector
from .local_max_peak_finder import LocalMaxPeakFinder
from .trackpy_local_max_peak_finder import TrackpyLocalMaxPeakFinder

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
__all__ = list(set(
    implementation_name
    for implementation_name, implementation_cls in locals().items()
    if isinstance(implementation_cls, type) and issubclass(implementation_cls, FindSpotsAlgorithm)
))
