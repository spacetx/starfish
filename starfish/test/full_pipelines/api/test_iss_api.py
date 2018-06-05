from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.pipeline.filter.white_tophat import WhiteTophat
from starfish.pipeline.registration.fourier_shift import FourierShiftRegistration
from starfish.test.dataset_fixtures import labeled_synthetic_dataset


def test_merfish_pipeline():
    stack = labeled_synthetic_dataset()

    fsr = FourierShiftRegistration(upsampling=1000)
    fsr.register(stack)

    wth = WhiteTophat(disk_size=15)
    wth.filter(stack)

    gsd = GaussianSpotDetector(blobs_image_name='dots', min_sigma=2, max_sigma=10, num_sigma=10, threshold=0.1)
    spot_attributes, encoded_spots = gsd.find(stack)

    assert spot_attributes.data.shape[0] == 19
