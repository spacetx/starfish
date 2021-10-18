"""
Tests for codebook.per_round_max_decode method
"""

import numpy as np
import pandas as pd
import pytest

from starfish import IntensityTable
from starfish.core.types import Axes, Features, SpotAttributes
from ..codebook import Codebook


def intensity_table_factory(data: np.ndarray = np.array([[[0, 3], [4, 0]]])) -> IntensityTable:
    """IntensityTable with a single feature that was measured over 2 channels and 2 rounds."""

    # generates spot attributes equal in size to the number of passed features.
    # each attribute has coordinates (z, y, x) equal to the feature index, and radius 1.
    spot_attributes_data = pd.DataFrame(
        data=np.array([[i, i, i, 1] for i in np.arange(data.shape[0])]),
        columns=[Axes.ZPLANE, Axes.Y, Axes.X, Features.SPOT_RADIUS]
    )

    spot_attributes = SpotAttributes(spot_attributes_data)
    intensity_table = IntensityTable.from_spot_data(
        data, spot_attributes,
        round_values=np.arange(data.shape[1]),
        ch_values=np.arange(data.shape[2]),
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
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: 'GENE_B'
        },
    ]
    return Codebook.from_code_array(codebook_array)


def test_intensity_tables_with_different_numbers_of_codes_or_channels_throw_value_error():
    """
    The test passes a 3-round and 1-round IntensityTable to a 2-round codebook.
    Both should raise a ValueError.
    Finally passes a valid 2-round IntensityTable which should not raise an error.
    """
    data = np.array(
        [[[4, 4],
          [3, 0],
          [1, 2]]]
    )
    codebook = codebook_factory()
    intensities = intensity_table_factory(data)

    with pytest.raises(ValueError, match="Codebook and Intensities must have same number"):
        codebook.decode_per_round_max(intensities)

    with pytest.raises(ValueError, match="Codebook and Intensities must have same number"):
        codebook.decode_per_round_max(intensities.sel(r=1))

    _ = codebook.decode_per_round_max(intensities.sel(r=slice(0, 1)))


def test_example_intensity_decodes_to_gene_a():
    """The single intensity should decode to gene a. Verify that it does."""
    intensities = intensity_table_factory()
    codebook = codebook_factory()

    decoded_intensities = codebook.decode_per_round_max(intensities)

    assert decoded_intensities[Features.TARGET] == 'GENE_A'
    assert np.array_equal(decoded_intensities[Features.PASSES_THRESHOLDS], [True])
    assert np.array_equal(decoded_intensities[Features.DISTANCE], [0])


def test_example_intensity_that_has_no_match_is_assigned_nan():
    """
    The decoder assigns nan when it doesn't receive a match, but the nan is converted to a
    string due to the object dtype
    """
    no_match = np.array([[[3, 0], [0, 4]]])
    intensities = intensity_table_factory(no_match)
    codebook = codebook_factory()

    decoded_intensities = codebook.decode_per_round_max(intensities)

    # even though the feature did not decode, it should still be returned
    assert decoded_intensities.sizes[Features.AXIS] == 1

    # assert that the feature did not decode
    assert decoded_intensities[Features.TARGET] == 'nan'
    assert np.array_equal(decoded_intensities[Features.PASSES_THRESHOLDS], [False])

    # distance is calculated as the fraction of signal NOT in the selected channel. Here all of
    # the signal is in the max channel, so distance is 0.
    # Admittedly, this is a little bit confusing, since distance here measures both how correct
    # a code is, but also how wrong one is when it _doesn't_ decode properly.
    assert np.array_equal(decoded_intensities[Features.DISTANCE], [0])


def test_argmax_selects_the_last_equal_intensity_channel_and_decodes_consistently():
    """
    When two channels are tied, argmax selects the last first one. This is the code for Gene A:

    [[0, 1],
     [1, 0]]

    both of these observations could potentially match the code for GENE_A:

    [[4, 3],
     [4, 0]]

    and

    [[0, 3],
     [4, 3]]

    The first is interpreted as argmax as:

    [[1, 1],
     [0, 0]]

    which fails to match, while the second is interpreted as

    [[0, 1],
     [1, 0]]

    and matches GENE_A.

    This test also verifies that the decoder can operate on multiple codes.
    """

    data = np.array(
        [[[4, 3],  # this code is decoded "wrong"
          [4, 0]],
         [[0, 3],  # this code is decoded "right"
          [4, 3]]]
    )
    intensities = intensity_table_factory(data)
    codebook = codebook_factory()

    decoded_intensities = codebook.decode_per_round_max(intensities)
    assert np.array_equal(decoded_intensities[Features.TARGET].values, ['nan', 'GENE_A'])


def test_argmax_does_not_select_first_code():
    """
    When all the channels in a round are uniform, argmax erroneously picks the first channel as the
    max.  In this case, it incorrectly assigns the wrong code for that round.  This test ensures
    that the workaround we put in for this works correctly.
    """

    data = np.array(
        [[[0.0, 1.0],
          [1.0, 1.0]],  # this round is uniform, so it will erroneously be decoded as the first ch.
         [[0.0, 1.0],
          [1.0, 0.0]]]
    )
    intensities = intensity_table_factory(data)
    codebook = codebook_factory()

    decoded_intensities = codebook.decode_per_round_max(intensities)
    assert np.array_equal(decoded_intensities[Features.TARGET].values, ['nan', 'GENE_A'])


def test_feature_round_all_nan():
    """
    When all the channels in a round are NaN, argmax chokes.  This test ensures that the workaround
    we put in for this works correctly.
    """

    data = np.array(
        [[[0.0, 1.0],
          [np.nan, np.nan]],
         [[0.0, 1.0],
          [1.0, 0.0]]]
    )
    intensities = intensity_table_factory(data)
    codebook = codebook_factory()

    decoded_intensities = codebook.decode_per_round_max(intensities)
    assert np.array_equal(decoded_intensities[Features.TARGET].values, ['nan', 'GENE_A'])
