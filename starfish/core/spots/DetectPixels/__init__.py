from ._base import DetectPixelsAlgorithm
from .pixel_spot_decoder import PixelSpotDecoder

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
all_filters = {
    filter_name: filter_cls
    for filter_name, filter_cls in locals().items()
    if isinstance(filter_cls, type) and issubclass(filter_cls, DetectPixelsAlgorithm)
}
__all__ = list(all_filters.keys())
