import numpy as np
import pytest

from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector

from starfish.util.synthesize import _synthetic_spots
from starfish.pipeline.features.codebook import Codebook


@pytest.mark.skip('needs codebook and data generated synthetically')
def test_intensity_table():
    data, codebook = _synthetic_spots()
    gsd = GaussianSpotDetector(blobs_image_name='dots', min_sigma=2, max_sigma=10, num_sigma=10, threshold=0.1)
    intensity_table = gsd.find(data)
    codebook = Codebook.from_code_array(codebook, 4, 4)
    result = codebook.euclidean_decode(intensity_table)

    expected_gene_decoding = np.array([77, 87, 27, 77, 74, 94, 106, 77, 32, 110, 21, 46, 103, 39, 85, 108, 11, 4])
    assert np.array_equal(result.indexes['features'].get_level_values('gene'), expected_gene_decoding)



