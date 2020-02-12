"""
Tests for codebook.from_code_array method
"""

from typing import Any, Dict, List

import numpy as np
import pytest

from starfish.core.types import Axes, Features
from ..codebook import Codebook


def codebook_array_factory() -> List[Dict[str, Any]]:
    """
    Codebook with two codewords describing an experiment with three channels and two imaging rounds.
    Both codes have two "on" channels.
    """
    return [
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


def assert_sizes(codebook, check_values=True):

    assert codebook.sizes[Axes.CH] == 3
    assert codebook.sizes[Axes.ROUND] == 2
    assert codebook.sizes[Features.TARGET] == 2

    if not check_values:
        return

    # codebook should have 4 "on" combinations
    expected_values = np.zeros((2, 2, 3))
    expected_values[0, 0, 0] = 1
    expected_values[0, 1, 1] = 1
    expected_values[1, 0, 2] = 1
    expected_values[1, 1, 1] = 1

    assert np.array_equal(codebook.values, expected_values)


def test_from_code_array_has_three_channels_two_rounds_and_two_codes():
    """
    Tests that from_code_array loads a small codebook that has the correct size and values
    """
    code_array: List = codebook_array_factory()
    codebook: Codebook = Codebook.from_code_array(code_array)
    assert_sizes(codebook)


def test_from_code_array_throws_key_error_with_missing_channel_round_or_value():
    """Tests that from_code_array throws errors when it encounters malformed codes"""
    code_array: List = codebook_array_factory()

    # codebook is now missing a round
    del code_array[0][Features.CODEWORD][0][Axes.ROUND.value]
    with pytest.raises(ValueError):
        Codebook.from_code_array(code_array)

    # codebook is now missing a ch
    code_array: List = codebook_array_factory()
    del code_array[0][Features.CODEWORD][0][Axes.CH.value]
    with pytest.raises(ValueError):
        Codebook.from_code_array(code_array)

    # codebook is now missing a value
    code_array: List = codebook_array_factory()
    del code_array[0][Features.CODEWORD][0][Features.CODE_VALUE]
    with pytest.raises(ValueError):
        Codebook.from_code_array(code_array)


def test_from_code_array_expands_codebook_when_provided_n_codes_that_exceeds_array_value():
    """
    The codebook factory produces codes with 3 channels and 2 rounds. This test provides numbers
    larger than that, and the codebook should be expanded to those numbers as a result.
    """
    code_array: List = codebook_array_factory()
    codebook: Codebook = Codebook.from_code_array(code_array, n_channel=10, n_round=4)
    assert codebook.sizes[Axes.CH] == 10
    assert codebook.sizes[Axes.ROUND] == 4
    assert codebook.sizes[Features.TARGET] == 2


def test_from_code_array_throws_exceptions_when_data_does_not_match_channel_or_round_requests():
    """
    The codebook factory produces codes with 3 channels and 2 rounds. This test provides numbers
    larger than that, and the codebook should be expanded to those numbers as a result.
    """
    code_array: List = codebook_array_factory()

    # should throw an exception, as 3 channels are present in the data
    with pytest.raises(ValueError):
        Codebook.from_code_array(code_array, n_channel=2, n_round=4)

    # should throw an exception, as 2 rounds are present in the data
    with pytest.raises(ValueError):
        Codebook.from_code_array(code_array, n_channel=3, n_round=1)


def test_from_code_array_throws_exception_when_data_is_improperly_formatted():
    code_array: List = codebook_array_factory()
    code_array[0][Features.CODEWORD][0] = ('I should be a dict, oops!',)
    with pytest.raises(TypeError):
        Codebook.from_code_array(code_array, n_channel=3, n_round=1)


# TODO codebook should throw an error when an empty array is passed


#
# Underlying methods
#

def test_empty_codebook():
    code_array: List = codebook_array_factory()
    targets = [x[Features.TARGET] for x in code_array]
    codebook = Codebook.zeros(targets, n_round=2, n_channel=3)
    assert_sizes(codebook, False)

def test_create_codebook():
    code_array: List = codebook_array_factory()
    targets = [x[Features.TARGET] for x in code_array]

    # Loop performed by from_code_array
    data = np.zeros((2, 2, 3), dtype=np.uint8)
    for i, code_dict in enumerate(code_array):
        for bit in code_dict[Features.CODEWORD]:
            ch = int(bit[Axes.CH])
            r = int(bit[Axes.ROUND])
            data[i, r, ch] = int(bit[Features.CODE_VALUE])

    codebook = Codebook.from_numpy(targets, n_round=2, n_channel=3, data=data)
    assert_sizes(codebook)
