import pytest

from starfish.constants import Indices
from starfish.test.dataset_fixtures import merfish_stack
from starfish.pipeline.features.pixels.pixel_spot_detector import PixelSpotDetector
from starfish.pipeline.filter.gaussian_high_pass import GaussianHighPass
from starfish.pipeline.filter.gaussian_low_pass import GaussianLowPass
from starfish.pipeline.filter.richardson_lucy_deconvolution import DeconvolvePSF


@pytest.mark.skip("TODO ambrosejcarr: fix this test in a future PR. Currently no spots are generated, and this fails.")
def test_merfish_pipeline(merfish_stack):
    s = merfish_stack

    # high pass filter
    ghp = GaussianHighPass(sigma=3)
    ghp.filter(s)

    # deconvolve the point spread function
    dpsf = DeconvolvePSF(num_iter=15, sigma=2)
    dpsf.filter(s)

    # low pass filter
    glp = GaussianLowPass(sigma=1)
    glp.filter(s)

    # scale the data by the scale factors
    scale_factors = {(t[Indices.ROUND], t[Indices.CH]): t['scale_factor'] for index, t in s.tile_metadata.iterrows()}
    for indices in s.image._iter_indices():
        data = s.image.get_slice(indices)[0]
        scaled = data / scale_factors[indices[Indices.ROUND], indices[Indices.CH]]
        s.image.set_slice(indices, scaled)

    # detect and decode spots
    psd = PixelSpotDetector(
        codebook='https://s3.amazonaws.com/czi.starfish.data.public/MERFISH/codebook.csv',
        distance_threshold=0.5176,
        magnitude_threshold=1,
        area_threshold=2,
        crop_size=40
    )

    psd.find(s)
