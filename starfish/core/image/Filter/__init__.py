from starfish.core.pipeline import import_all_submodules
# from ._base import Filter
# import_all_submodules(__file__, __package__)
from .clip import Clip
from .bandpass import Bandpass
from .clip_percentile_to_zero import ClipPercentileToZero
from .clip_value_to_zero import ClipValueToZero
from .element_wise_mult import ElementWiseMultiply
from .gaussian_high_pass import GaussianHighPass
from .gaussian_low_pass import GaussianLowPass
from .laplace import Laplace
from .linear_unmixing import LinearUnmixing
from .max_proj import MaxProject
from .match_histograms import MatchHistograms
from .mean_high_pass import MeanHighPass
from .reduce import Reduce
from .richardson_lucy_deconvolution import DeconvolvePSF
from .white_tophat import WhiteTophat
from .zero_by_channel_magnitude import ZeroByChannelMagnitude
