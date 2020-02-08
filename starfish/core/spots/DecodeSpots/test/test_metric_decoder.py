import numpy as np

from starfish.core.image import Filter
from starfish.core.spots import DecodeSpots
from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match
from starfish.core.spots.FindSpots.blob import BlobDetector
from starfish.core.test.factories import SyntheticData
from starfish.core.types import Axes, FunctionSource


def test_metric_decoder_original_intensities_flag():
    np.random.seed(0)

    n_z = 40
    height = 300
    width = 400
    sigma = 2

    sd = SyntheticData(
        n_round=4,
        n_ch=4,
        n_z=n_z,
        height=height,
        width=width,
        n_spots=100,
        n_codes=10,
        point_spread_function=(sigma, sigma, sigma),
    )

    codebook = sd.codebook()
    intensities = sd.intensities(codebook=codebook)

    spots = sd.spots(intensities=intensities)
    gsd = BlobDetector(min_sigma=1, max_sigma=4, num_sigma=5, threshold=1e-4)
    spots_max_projector = Filter.Reduce((Axes.ROUND, Axes.CH), func=FunctionSource.np("max"))
    spots_max = spots_max_projector.run(spots)
    spot_results = gsd.run(image_stack=spots, reference_image=spots_max)

    og_intensities = build_spot_traces_exact_match(spot_results=spot_results)

    decoder = DecodeSpots.MetricDistance(codebook=codebook, max_distance=1,
                                         min_intensity=0, norm_order=2,
                                         return_original_intensities=True)
    decoded_intensities = decoder.run(spots=spot_results)

    # assert that original intensities were returned in the decocded intensity file
    assert np.array_equal(decoded_intensities.values, og_intensities.values)

    # now run without flag, assert intensities change
    decoder = DecodeSpots.MetricDistance(codebook=codebook, max_distance=1,
                                         min_intensity=0, norm_order=2,
                                         return_original_intensities=False)
    decoded_intensities = decoder.run(spots=spot_results)

    assert not np.array_equal(decoded_intensities.values, og_intensities.values)
