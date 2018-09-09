"""
Tests for codebook.from_code_array method
"""

from typing import Any, Dict, List

import numpy as np
import pytest

from starfish import Codebook
from starfish.types import Features, Indices


def codebook_array_factory() -> List[Dict[str, Any]]:
    """
    Codebook with two codewords describing an experiment with three channels and two imaging rounds.
    Both codes have two "on" channels.
    """
    return [
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


def test_from_code_array_has_three_channels_two_rounds_and_two_codes():
    """
    Tests that from_code_array loads a small codebook that has the correct size and values
    """
    code_array: List = codebook_array_factory()
    codebook: Codebook = Codebook.from_code_array(code_array)

    assert codebook.sizes[Indices.CH] == 3
    assert codebook.sizes[Indices.ROUND] == 2
    assert codebook.sizes[Features.TARGET] == 2

    # codebook should have 4 "on" combinations
    expected_values = np.zeros((2, 3, 2))
    expected_values[0, 0, 0] = 1
    expected_values[0, 1, 1] = 1
    expected_values[1, 2, 0] = 1
    expected_values[1, 1, 1] = 1

    assert np.array_equal(codebook.values, expected_values)


# TODO ambrosejcarr: this should be a ValueError, not a KeyError,
# and the message should be clearer to the user
def test_from_code_array_throws_key_error_with_missing_channel_round_or_value():
    """Tests that from_code_array throws errors when it encounters malformed codes"""
    code_array: List = codebook_array_factory()

    # codebook is now missing a channel
    del code_array[0][Features.CODEWORD][0][Indices.ROUND.value]
    with pytest.raises(KeyError):
        Codebook.from_code_array(code_array)

    code_array: List = codebook_array_factory()
    del code_array[0][Features.CODEWORD][0][Indices.CH.value]
    with pytest.raises(KeyError):
        Codebook.from_code_array(code_array)

    code_array: List = codebook_array_factory()
    del code_array[0][Features.CODEWORD][0][Features.CODE_VALUE]
    with pytest.raises(KeyError):
        Codebook.from_code_array(code_array)


def test_from_code_array_expands_codebook_when_provided_n_codes_that_exceeds_array_value():
    """
    The codebook factory produces codes with 3 channels and 2 rounds. This test provides numbers
    larger than that, and the codebook should be expanded to those numbers as a result.
    """
    code_array: List = codebook_array_factory()
    codebook: Codebook = Codebook.from_code_array(code_array, n_ch=10, n_round=4)
    assert codebook.sizes[Indices.CH] == 10
    assert codebook.sizes[Indices.ROUND] == 4
    assert codebook.sizes[Features.TARGET] == 2


def test_from_code_array_throws_exceptions_when_data_does_not_match_channel_or_round_requests():
    """
    The codebook factory produces codes with 3 channels and 2 rounds. This test provides numbers
    larger than that, and the codebook should be expanded to those numbers as a result.
    """
    code_array: List = codebook_array_factory()

    # should throw an exception, as 3 channels are present in the data
    with pytest.raises(ValueError):
        Codebook.from_code_array(code_array, n_ch=2, n_round=4)

    # should throw an exception, as 2 rounds are present in the data
    with pytest.raises(ValueError):
        Codebook.from_code_array(code_array, n_ch=3, n_round=1)


def test_from_code_array_throws_exception_when_data_is_improperly_formatted():
    code_array: List = codebook_array_factory()
    code_array[0][Features.CODEWORD][0] = ('I should be a dict, oops!',)
    with pytest.raises(TypeError):
        Codebook.from_code_array(code_array, n_ch=3, n_round=1)


# TODO codebook should throw an error when an empty array is passed
