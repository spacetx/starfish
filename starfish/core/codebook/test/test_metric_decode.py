"""
Tests for codebook.metric_decode method
"""

import numpy as np
import pandas as pd
import pytest

from starfish import IntensityTable
from starfish.core.types import Axes, Features, SpotAttributes
from ..codebook import Codebook


def intensity_table_factory(data: np.ndarray=np.array([[[0, 3], [4, 0]]])) -> IntensityTable:
    """
    Produces an IntensityTable with a single feature that was measured over 2 channels and 2 rounds.
    """

    # generates spot attributes equal in size to the number of passed features.
    # each attribute has coordinates (z, y, x) equal to the feature index, and radius 1.
    spot_attributes_data = pd.DataFrame(
        data=np.array([[i, i, i, 1] for i in np.arange(data.shape[0])]),
        columns=[Axes.ZPLANE, Axes.Y, Axes.X, Features.SPOT_RADIUS]
    )

    intensity_table = IntensityTable.from_spot_data(
        data,
        SpotAttributes(spot_attributes_data),
        ch_values=np.arange(data.shape[1]),
        round_values=np.arange(data.shape[2]),
    )
    return intensity_table


def codebook_factory() -> Codebook:
    """
    Codebook with two codewords describing an experiment with three channels and two imaging rounds.
    Both codes have two "on" channels.
    """
    codebook_array = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: 'GENE_A'
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 1, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: 'GENE_B'
        },
    ]
    return Codebook.from_code_array(codebook_array)


def test_metric_decode():
    """
    This test exposes 3 test features, each the the same normalized trace.
    The first should decode to GENE_A, and pass both the intensity and distance filters
    The second should decode to GENE_B, but fail the intensity filter
    The third should decode to GENE_B, as it is less far from that gene than GENE_A, but
    should nevertheless fail the distance filter because the tiles other than (0, 0) don't
    match
    """
    data = np.array(
        [[[0, 4],  # this code is decoded "right"
          [3, 0]],
         [[0, 0],  # this code should be filtered based on magnitude
          [0.4, 0.3]],
         [[30, 0],  # this code should be filtered based on distance
          [0, 40]]]
    )
    intensities = intensity_table_factory(data)
    codebook = codebook_factory()

    decoded_intensities = codebook.decode_metric(
        intensities,
        max_distance=0.5,
        min_intensity=1,
        norm_order=1
    )

    assert hasattr(decoded_intensities, Features.DISTANCE)

    assert decoded_intensities.sizes[Features.AXIS] == 3

    assert np.array_equal(
        decoded_intensities[Features.TARGET].values,
        ['GENE_A', 'GENE_B', 'GENE_B'],
    )

    assert np.array_equal(
        decoded_intensities[Features.PASSES_THRESHOLDS].values,
        [True, False, False]
    )

    assert not np.all(decoded_intensities == intensities)


def test_unmatched_intensities_and_codebook_table_sizes_throws_value_error():
    """
    Codebook and Intensity channel and round number must match. Here we use a codebook with 3
    channels, but an IntensityTable with only 2 to verify an error is thrown.
    """

    # this codebook has 3 channels
    codebook_array = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 2, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: 'GENE_A'
        },
    ]
    codebook = Codebook.from_code_array(codebook_array)
    intensities = intensity_table_factory()
    with pytest.raises(ValueError):
        codebook.decode_metric(intensities, max_distance=0.5, min_intensity=1, norm_order=1)
