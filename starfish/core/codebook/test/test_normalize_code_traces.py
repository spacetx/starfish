import numpy as np
import pandas as pd
import pytest

from starfish import IntensityTable
from starfish.core.types import Axes, Features, SpotAttributes
from ..codebook import Codebook


def intensity_table_factory() -> IntensityTable:
    """IntensityTable with a single feature that was measured over 2 channels and 2 rounds."""

    intensities = np.array(
        [[[0, 3],
          [4, 0]]],
        dtype=float
    )
    spot_attribute_data = pd.DataFrame(
        data=[0, 0, 0, 1],
        index=[Axes.ZPLANE.value, Axes.Y.value, Axes.X.value, Features.SPOT_RADIUS]
    ).T
    spot_attributes = SpotAttributes(spot_attribute_data)

    intensity_table = IntensityTable.from_spot_data(
        intensities, spot_attributes,
        ch_values=np.arange(intensities.shape[1]),
        round_values=np.arange(intensities.shape[2]),
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
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 2, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    return Codebook.from_code_array(codebook_array)


@pytest.mark.parametrize('norm_order, expected_size', [
    (1, 2),
    (2, np.sqrt(2))
])
def test_normalize_codes(norm_order, expected_size):
    """
    Create a simple codebook with two features, each with two "on" sites. For these simple cases,
    we can easily calculate the expected value of the norms. (2-norm: sqrt 2, 1-norm: 2), Verify
    that the output of the functions correspond to these expectations.
    """
    codebook = codebook_factory()
    normed_codebook, norms = Codebook._normalize_features(codebook, norm_order)
    assert np.all(norms == expected_size)

    # each code should still have only two non-zero values
    assert np.all(
        normed_codebook.groupby(Features.TARGET).map(lambda x: np.sum(x != 0)) == 2
    )

    # each non-zero value should be equal to 1 / expected_size of the norm. There are two non-zero
    # values and so the sum of the code should be (1 / expected_size) * 2
    assert np.all(
        normed_codebook.sum((Axes.CH.value, Axes.ROUND.value)) == (1 / expected_size) * 2
    )


@pytest.mark.parametrize('norm_order, expected_size', [
    (1, 7),
    (2, 5)
])
def test_normalize_intensities(norm_order, expected_size):
    """
    Create a slightly less simple IntensityTable with one "on" tile per feature,
    we can again calculate the expected value of the norms. Verify
    that the output of the functions correspond to these expectations.
    """
    intensity_table = intensity_table_factory()
    normed_intensities, norms = Codebook._normalize_features(intensity_table, norm_order)

    assert np.all(norms == expected_size)

    # each feature should still have only two non-zero values
    assert np.all(
        normed_intensities.groupby(Features.AXIS).map(lambda x: np.sum(x != 0)) == 2
    )

    # each non-zero value should be equal to 1 / expected_size of the norm.
    assert np.all(normed_intensities == intensity_table / norms)


# TODO I think the outcome of this test should be NaNs -- it's too sensitive to boundary conditions
# 0/0/0/0 vs 0/0/0/1e-30 give VERY different results
@pytest.mark.skip('Test is wrong, needs to be revisited')
@pytest.mark.parametrize('norm_order, expected_value', [
    (1, 1 / 4),
    (2, 1 / 8)
])
def test_all_blank_features_yield_non_zero_but_equal_normalized_values(norm_order, expected_value):

    intensity_table = intensity_table_factory()

    # zero-out all the values in the IntensityTable
    intensity_table.values = np.zeros(4).reshape(1, 2, 2)
    normed_intensities, norms = Codebook._normalize_features(intensity_table, norm_order)

    # todo norms here are zero, which seems like the right answer!
    assert norms == 0

    assert np.all(normed_intensities == expected_value)
