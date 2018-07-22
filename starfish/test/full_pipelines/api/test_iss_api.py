from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.pipeline.filter.white_tophat import WhiteTophat
from starfish.pipeline.registration.fourier_shift import FourierShiftRegistration
from starfish.util.synthesize import SyntheticData
import numpy as np
from starfish.constants import Indices
from starfish.image import ImageStack


def test_iss_pipeline():
    np.random.seed(2)
    synthesizer = SyntheticData(n_spots=5)
    codebook = synthesizer.codebook()
    true_intensities = synthesizer.intensities(codebook=codebook)
    image = synthesizer.spots(intensities=true_intensities)

    dots_data = image.max_proj(Indices.ROUND, Indices.CH, Indices.Z)
    dots = ImageStack.from_numpy_array(dots_data.reshape((1, 1, 1, *dots_data.shape)))

    wth = WhiteTophat(disk_size=15)
    wth.filter(image)
    wth.filter(dots)

    fsr = FourierShiftRegistration(upsampling=1000, reference_stack=dots)
    fsr.register(image)

    min_sigma = 1.5
    max_sigma = 5
    num_sigma = 10
    threshold = 1e-4
    gsd = GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        blobs_stack=dots,
        measurement_type='max',
    )

    intensities = gsd.find(image_stack=image)
    assert intensities.shape[0] == 5

    codebook.metric_decode(intensities, max_distance=1, min_intensity=0, norm_order=2)
