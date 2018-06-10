import numpy as np

from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector

from starfish.util.synthesize import synthesize
from starfish.pipeline.features.codebook import Codebook


def test_intensity_table():
    data, codebook = synthesize(num_ch=4)
    gsd = GaussianSpotDetector(blobs_image_name='dots', min_sigma=2, max_sigma=10, num_sigma=10, threshold=0.1)
    intensity_table = gsd.find(data)
    codebook = Codebook.from_code_array(codebook, 4, 4)
    result = codebook.decode(intensity_table)

    expected_gene_decoding = np.array([77, 87, 27, 77, 74, 94, 106, 77, 32, 110, 21, 46, 103, 39, 85, 108, 11, 4])
    assert np.array_equal(result.indexes['features'].get_level_values('gene'), expected_gene_decoding)



