import pytest

from starfish.constants import Indices
from starfish.spots._detector.pixel_spot_detector import PixelSpotDetector
from starfish.image._filter.gaussian_high_pass import GaussianHighPass
from starfish.image._filter.gaussian_low_pass import GaussianLowPass
from starfish.image._filter.richardson_lucy_deconvolution import DeconvolvePSF


@pytest.mark.skip("TODO ambrosejcarr: fix this test in a future PR. Currently no spots are "
                  "generated, and this fails.")
def test_merfish_pipeline(merfish_stack):
    s = merfish_stack

    # high pass filter
    ghp = GaussianHighPass(sigma=3)
    ghp.run(s)

    # deconvolve the point spread function
    dpsf = DeconvolvePSF(num_iter=15, sigma=2)
    dpsf.run(s)

    # low pass filter
    glp = GaussianLowPass(sigma=1)
    glp.run(s)

    # scale the data by the scale factors
    scale_factors = {(t[Indices.ROUND], t[Indices.CH]): t['scale_factor']
                     for index, t in s.tile_metadata.iterrows()}
    for indices in s.image._iter_indices():
        data = s.image.get_slice(indices)[0]
        scaled = data / scale_factors[indices[Indices.ROUND], indices[Indices.CH]]
        s.image.set_slice(indices, scaled)

    # detect and decode spots
    psd = PixelSpotDetector(
        codebook='https://dmf0bdeheu4zf.cloudfront.net/MERFISH/codebook.csv',
        metric='euclidean',
        distance_threshold=0.5176,
        magnitude_threshold=1,
        area_threshold=2,
        crop_size=40
    )

    psd.find(s)
