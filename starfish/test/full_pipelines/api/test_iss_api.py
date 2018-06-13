from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.pipeline.filter.white_tophat import WhiteTophat
from starfish.pipeline.registration.fourier_shift import FourierShiftRegistration
from starfish.test.dataset_fixtures import labeled_synthetic_dataset
from starfish.pipeline.features.codebook import Codebook


def test_iss_pipeline():
    stack, code_array = labeled_synthetic_dataset()

    fsr = FourierShiftRegistration(upsampling=1000)
    fsr.register(stack)

    wth = WhiteTophat(disk_size=15)
    wth.filter(stack)

    gsd = GaussianSpotDetector(blobs_image_name='dots', min_sigma=2, max_sigma=10, num_sigma=10, threshold=0.1)

    intesities = gsd.find(stack)
    codebook = Codebook.from_code_array(code_array, n_ch=4, n_hyb=4)
    intesities = codebook.euclidean_decode(intesities)

    assert intesities.shape[0] == 19
