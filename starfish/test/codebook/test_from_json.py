"""
Tests for codebook.from_json method
"""

import json
import os
import tempfile
from typing import Any, Dict, List

from starfish import Codebook
from starfish.types import Features, Indices


def codebook_json_data_factory() -> List[Dict[str, Any]]:
    """
    Codebook with two codewords describing an experiment with three channels and two imaging rounds.
    Both codes have two "on" channels.
    """
    codebook_data = [
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
    return codebook_data


def test_codebook_loads_from_local_file() -> None:
    """
    dumps the codebook data to a temporary json file and reads it back into a Codebook,
    verifying that the data has not changed.
    """

    # dump codebook to disk
    codebook_data: List = codebook_json_data_factory()
    with tempfile.TemporaryDirectory() as directory:
        codebook_json: str = os.path.join(directory, 'simple_codebook.json')
        with open(codebook_json, 'w') as f:
            json.dump(codebook_data, f)

        # load the codebook
        codebook = Codebook.from_json(codebook_json)
        assert codebook.sizes[Indices.ROUND] == 2
        assert codebook.sizes[Indices.CH] == 3
        assert codebook.sizes[Features.TARGET] == 2


def test_codebook_serialization():
    """
    Test that codebook can be saved to disk and recovered, and that the recovered codebook is
    identical to the one that it was serialized from.
    """
    # Create a codebook
    codebook_array = codebook_json_data_factory()
    codebook = Codebook.from_code_array(codebook_array)

    # Dump it to a temporary file
    with tempfile.TemporaryDirectory() as directory:
        json_codebook = os.path.join(directory, 'codebook.json')
        codebook.to_json(json_codebook)

        # Retrieve it and test that the data it contains has not changed
        codebook_reloaded = Codebook.from_json(json_codebook)
        assert codebook_reloaded.equals(codebook)
