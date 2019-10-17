"""
Tests for codebook.code_length method
"""

from .test_from_code_array import codebook_array_factory
from ..codebook import Codebook


def test_code_length_properly_counts_bit_length_of_codes():
    """
    use the codebook factory from another testing module that produces codes with 3 channels and
    two rounds. This produces a code which should have length 6.
    """
    codebook_data = codebook_array_factory()
    codebook = Codebook.from_code_array(codebook_data)
    assert codebook.code_length == 6
