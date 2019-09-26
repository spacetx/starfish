import numpy as np
import pytest

import starfish.data
from starfish import FieldOfView
from starfish.image import Filter
from starfish.spots import FindSpots
from starfish.types import TraceBuildingStrategies


@pytest.mark.skip('This test runs but takes forever')
def test_allen_smFISH_cropped_data():

    # set random seed to errors provoked by optimization functions
    np.random.seed(777)

    # load the experiment
    experiment = starfish.data.allen_smFISH(use_test_data=True)

    primary_image = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)

    clip = Filter.Clip(p_min=10, p_max=100)
    clipped_image = clip.run(primary_image, in_place=False)

    bandpass = Filter.Bandpass(lshort=0.5, llong=7, threshold=0.0, truncate=4)
    bandpassed_image = bandpass.run(clipped_image, in_place=False)

    clip = Filter.Clip(p_min=10, p_max=100, is_volume=False)
    clipped_bandpassed_image = clip.run(bandpassed_image, in_place=False)

    sigma = (1, 0, 0)  # filter only in z, do nothing in x, y
    glp = Filter.GaussianLowPass(sigma=sigma, is_volume=True)
    z_filtered_image = glp.run(clipped_bandpassed_image, in_place=False)

    tlmpf = FindSpots.TrackpyLocalMaxPeakFinder(
        spot_diameter=5,  # must be odd integer
        min_mass=0.02,
        max_size=2,  # this is max radius
        separation=7,
        noise_size=0.65,  # this is not used because preprocess is False
        preprocess=False,
        percentile=10,
        # this is irrelevant when min_mass, spot_diameter, and max_size are set properly
        verbose=True,
        is_volume=True,
    )
    spots = tlmpf.run(z_filtered_image)  # noqa

    decoder = starfish.spots.DecodeSpots.PerRoundMaxChannel(
        codebook=experiment.codebook,
        trace_building_strategy=TraceBuildingStrategies.SEQUENTIAL
    )
    decoder.run(spots=spots)
