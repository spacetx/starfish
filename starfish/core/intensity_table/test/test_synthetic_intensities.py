"""
Tests for IntensityTable.synthetic_intensities method.
"""

import numpy as np

from starfish.core.codebook.test.factories import codebook_array_factory
from starfish.core.types import Axes, Features
from ..intensity_table import IntensityTable


def test_synthetic_intensity_generation():
    """
    Create a 2-spot IntensityTable of pixel size (z=3, y=4, x=5) from a codebook with 3 channels
    and 2 rounds.

    Verify that the constructed Synthetic IntensityTable conforms to those dimensions, and given
    a known random seed, that the output spots decode to match a target in the input Codebook
    """
    # set seed to check that codebook is matched. This seed generates 2 instances of GENE_B
    np.random.seed(2)
    codebook = codebook_array_factory()
    num_z, height, width = 3, 4, 5
    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=num_z,
        height=height,
        width=width,
        n_spots=2,
    )

    # sizes should match codebook
    assert intensities.sizes[Axes.ROUND] == 2
    assert intensities.sizes[Axes.CH] == 3
    assert intensities.sizes[Features.AXIS] == 2

    # attributes should be bounded by the specified size
    assert np.all(intensities[Axes.ZPLANE.value] <= num_z)
    assert np.all(intensities[Axes.Y.value] <= height)
    assert np.all(intensities[Axes.X.value] <= width)

    # both codes should match GENE_B
    gene_b_intensities = codebook.sel(target="GENE_B")
    for feature_id in range(intensities.sizes[Features.AXIS]):
        feature_intensities = intensities[{Features.AXIS: feature_id}]
        assert np.array_equal(gene_b_intensities.values, feature_intensities.values)
