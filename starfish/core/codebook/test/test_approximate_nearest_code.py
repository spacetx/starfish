"""
Tests for codebook._approximate_nearest_code method
"""

import numpy as np

from .test_metric_decode import codebook_factory, intensity_table_factory


def test_simple_intensities_find_correct_nearest_code():
    """
    Test four simple examples for correct decoding. Here the first example should decode to GENE_A,
    the second to GENE_B. The third is closer to GENE_A. The fourth is equidistant to GENE_A and
    GENE_B, but it picks GENE_A because GENE_A comes first in the codebook.
    """
    data = np.array(
        [[[0, 0.5],
          [0.5, 0]],
         [[0, 0],
          [0.5, 0.5]],
         [[0.5, 0.5],
          [0, 0]],
         [[0, 0.5],
          [0, 0.5]]]
    )
    intensities = intensity_table_factory(data=data)
    codebook = codebook_factory()
    distances, gene_ids = codebook._approximate_nearest_code(
        codebook,
        intensities,
        metric='euclidean'
    )

    assert np.array_equal(gene_ids, ['GENE_A', 'GENE_B', 'GENE_A', 'GENE_A'])
