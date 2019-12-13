import numpy as np
import pytest

from starfish.core.image import Filter
from starfish.core.spots import DecodeSpots
from starfish.core.spots.FindSpots.blob import BlobDetector
from starfish.core.test.factories import SyntheticData
from starfish.core.types import Axes, Features, FunctionSource


def test_round_trip_synthetic_data():
    np.random.seed(0)

    sd = SyntheticData(
        n_ch=2,
        n_round=3,
        n_spots=1,
        n_codes=4,
        n_photons_background=0,
        background_electrons=0,
        camera_detection_efficiency=1.0,
        gray_level=1,
        ad_conversion_bits=16,
        point_spread_function=(2, 2, 2),
    )

    codebook = sd.codebook()
    intensities = sd.intensities(codebook=codebook)
    spots = sd.spots(intensities=intensities)
    spots_max_projector = Filter.Reduce((Axes.ROUND, Axes.CH), func=FunctionSource.np("max"))
    spots_max = spots_max_projector.run(spots)

    gsd = BlobDetector(min_sigma=1, max_sigma=4, num_sigma=5, threshold=0)
    spot_results = gsd.run(image_stack=spots, reference_image=spots_max)

    decoder = DecodeSpots.MetricDistance(codebook=codebook, max_distance=1,
                                         min_intensity=0, norm_order=2)
    decoded_intensities = decoder.run(spots=spot_results)

    # applying the gaussian blur to the intensities causes them to be reduced in magnitude, so
    # they won't be the same size, but they should be in the same place, and decode the same
    # way
    spot1, ch1, round1 = np.where(intensities.values)
    spot2, ch2, round2 = np.where(decoded_intensities.values)
    assert np.array_equal(spot1, spot2)
    assert np.array_equal(ch1, ch2)
    assert np.array_equal(round1, round2)
    assert len(decoded_intensities.coords[Features.TARGET]) == 1


@pytest.mark.slow
def test_medium_synthetic_stack():
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

    # some spots won't be detected properly because they will spread outside the image where
    # blurred,
    # so we'll remove those from intensities before we generate spots.

    spot_radius = sigma * np.sqrt(2)  # this is the radius of the spot in pixels

    valid_z = np.logical_and(intensities.z.values > spot_radius,
                             intensities.z.values < (n_z - spot_radius))
    valid_y = np.logical_and(intensities.y.values > spot_radius,
                             intensities.y.values < (height - spot_radius))
    valid_x = np.logical_and(intensities.x.values > spot_radius,
                             intensities.x.values < (width - spot_radius))

    valid_locations = valid_z & valid_y & valid_x
    intensities = intensities[np.where(valid_locations)]
    spots = sd.spots(intensities=intensities)
    gsd = BlobDetector(min_sigma=1, max_sigma=4, num_sigma=5, threshold=1e-4)
    spots_max_projector = Filter.Reduce((Axes.ROUND, Axes.CH), func=FunctionSource.np("max"))
    spots_max = spots_max_projector.run(spots)
    spot_results = gsd.run(image_stack=spots, reference_image=spots_max)

    decoder = DecodeSpots.MetricDistance(codebook=codebook, max_distance=1,
                                         min_intensity=0, norm_order=2)
    decoded_intensities = decoder.run(spots=spot_results)
    assert len(decoded_intensities.coords[Features.TARGET]) == 80
