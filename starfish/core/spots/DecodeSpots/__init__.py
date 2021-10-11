from ._base import DecodeSpotsAlgorithm
from .metric_decoder import MetricDistance
from .per_round_max_channel_decoder import PerRoundMaxChannel
from .simple_lookup_decoder import SimpleLookupDecoder
from .check_all_decoder import CheckAll

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
__all__ = list(set(
    implementation_name
    for implementation_name, implementation_cls in locals().items()
    if isinstance(implementation_cls, type) and issubclass(implementation_cls, DecodeSpotsAlgorithm)
))
