from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.pipeline.filter.white_tophat import WhiteTophat
from starfish.pipeline.registration.fourier_shift import FourierShiftRegistration
from starfish.pipeline.features.codebook import Codebook
from starfish.test.dataset_fixtures import gold_standard_dataset


# TODO ambrosejcarr debug synthetic data with GaussianSpotDetector
def test_iss_pipeline(gold_standard_dataset):

    stack, code_array = gold_standard_dataset

    fsr = FourierShiftRegistration(upsampling=1000)
    fsr.register(stack)

    wth = WhiteTophat(disk_size=15)
    wth.filter(stack)

    gsd = GaussianSpotDetector(blobs_image_name='dots', min_sigma=2, max_sigma=10, num_sigma=10, threshold=0.1)
    intesities = gsd.find(stack)
    codebook = Codebook.from_code_array(code_array, num_chs=4, num_hybs=4)
    intesities = codebook.decode(intesities)

    assert intesities.shape[0] == 19
