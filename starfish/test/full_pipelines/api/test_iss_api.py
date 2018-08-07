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

    wth = WhiteTophat(masking_radius=15)
    wth.run(image)
    wth.run(dots)

    fsr = FourierShiftRegistration(upsampling=1000, reference_stack=dots)
    fsr.run(image)

    min_sigma = 1.5
    max_sigma = 5
    num_sigma = 10
    threshold = 1e-4
    gsd = GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        measurement_type='max',
    )
    blobs_image = dots.numpy_array.reshape(1, *dots_data.shape)
    intensities = gsd.find(data_stack=image, blobs_image=blobs_image)
    assert intensities.shape[0] == 5

    codebook.metric_decode(intensities, max_distance=1, min_intensity=0, norm_order=2)
