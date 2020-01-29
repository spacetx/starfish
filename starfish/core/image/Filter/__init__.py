from ._base import FilterAlgorithm
from .bandpass import Bandpass
from .clip import Clip
from .clip_percentile_to_zero import ClipPercentileToZero
from .clip_value_to_zero import ClipValueToZero
from .element_wise_mult import ElementWiseMultiply
from .gaussian_high_pass import GaussianHighPass
from .gaussian_low_pass import GaussianLowPass
from .ilastik_pre_trained_probability import IlastikPretrainedProbability
from .laplace import Laplace
from .linear_unmixing import LinearUnmixing
from .map import Map
from .match_histograms import MatchHistograms
from .mean_high_pass import MeanHighPass
from .reduce import Reduce
from .richardson_lucy_deconvolution import DeconvolvePSF
from .white_tophat import WhiteTophat
from .zero_by_channel_magnitude import ZeroByChannelMagnitude

# autodoc's automodule directive only captures the modules explicitly listed in __all__.
__all__ = list(set(
    implementation_name
    for implementation_name, implementation_cls in locals().items()
    if isinstance(implementation_cls, type) and issubclass(implementation_cls, FilterAlgorithm)
))
