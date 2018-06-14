from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.pipeline.filter.white_tophat import WhiteTophat
from starfish.pipeline.registration.fourier_shift import FourierShiftRegistration
from starfish.io import Stack
import pytest
from starfish.test.dataset_fixtures import labeled_synthetic_dataset
from starfish.pipeline.features.codebook import Codebook


@pytest.mark.skip('needs a dots image to be created somehow')
def test_iss_pipeline(labeled_synthetic_dataset):
    image_stack = labeled_synthetic_dataset()
    stack = Stack.from_data(image_stack)

    fsr = FourierShiftRegistration(upsampling=1000)
    fsr.register(stack)

    wth = WhiteTophat(disk_size=15)
    wth.filter(stack)

    gsd = GaussianSpotDetector(blobs_image_name='dots', min_sigma=2, max_sigma=10, num_sigma=10, threshold=0.1)

    intesities = gsd.find(stack)
    # # codebook = Codebook.from_code_array(code_array, n_ch=4, n_hyb=4)
    # # intesities = codebook.euclidean_decode(intesities)
    #
    # assert intesities.shape[0] == 19
