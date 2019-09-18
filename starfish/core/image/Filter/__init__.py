from ._base import FilterAlgorithm
from .bandpass import Bandpass
from .clip import Clip
from .clip_percentile_to_zero import ClipPercentileToZero
from .clip_value_to_zero import ClipValueToZero
from .element_wise_mult import ElementWiseMultiply
from .gaussian_high_pass import GaussianHighPass
from .gaussian_low_pass import GaussianLowPass
from .laplace import Laplace
from .linear_unmixing import LinearUnmixing
from .map import Map
from .match_histograms import MatchHistograms
from .max_proj import MaxProject
from .mean_high_pass import MeanHighPass
from .reduce import Reduce
from .richardson_lucy_deconvolution import DeconvolvePSF
from .white_tophat import WhiteTophat
from .zero_by_channel_magnitude import ZeroByChannelMagnitude

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
all_filters = {
    filter_name: filter_cls
    for filter_name, filter_cls in locals().items()
    if isinstance(filter_cls, type) and issubclass(filter_cls, FilterAlgorithm)
}
__all__ = list(all_filters.keys())
