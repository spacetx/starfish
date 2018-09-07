"""
Tests for IntensityTable.synthetic_intensities method.
"""

import numpy as np

from starfish import Codebook, IntensityTable
from starfish.types import Features, Indices


def codebook_array_factory() -> Codebook:
    """
    Codebook with two codewords describing an experiment with three channels and two imaging rounds.
    Both codes have two "on" channels.
    """
    data = [
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 0, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 2, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    return Codebook.from_code_array(data)


def test_synthetic_intensity_generation():
    """
    Create a 2-spot IntensityTable of pixel size (z=3, y=4, x=5) from a codebook with 3 channels
    and 2 rounds.

    Verify that the constructed Synthetic IntensityTable conforms to those dimensions, and given
    a known random seed, that the output spots decode to match a target in the input Codebook
    """
    # set seed to check that codebook is matched. This seed generates 2 instances of GENE_B
    np.random.seed(1)
    codebook = codebook_array_factory()
    num_z, height, width = 3, 4, 5
    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=num_z,
        height=height,
        width=width,
        n_spots=2
    )

    # sizes should match codebook
    assert intensities.sizes[Indices.ROUND] == 2
    assert intensities.sizes[Indices.CH] == 3
    assert intensities.sizes[Features.AXIS] == 2

    # attributes should be bounded by the specified size
    assert np.all(intensities[Indices.Z.value] <= num_z)
    assert np.all(intensities[Indices.Y.value] <= height)
    assert np.all(intensities[Indices.X.value] <= width)

    # both codes should match GENE_B
    assert np.array_equal(
        np.where(intensities.values),
        [[0, 0, 1, 1],  # two each in feature 0 & 1
         [1, 2, 1, 2],  # one each in channel 1 & 2
         [1, 0, 1, 0]],  # channel 1 matches round 1, channel 2 matches round zero
    )
