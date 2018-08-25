import json
import os
import shutil
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

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
    """dumps the codebook data to a temporary json file and reads it back into a Codebook."""

    # dump codebook to disk
    codebook_data: List = codebook_json_data_factory()
    directory = tempfile.mkdtemp()
    codebook_json: str = os.path.join(directory, 'simple_codebook.json')
    with open(codebook_json, 'w') as f:
        json.dump(codebook_data, f)

    # load the codebook
    codebook = Codebook.from_json(codebook_json)

    assert codebook.sizes[Indices.ROUND] == 2
    assert codebook.sizes[Indices.CH] == 3
    assert codebook.sizes[Features.TARGET] == 2

    # clean up
    shutil.rmtree(directory)


@patch('starfish.codebook.urllib.request.urlopen')
def test_codebook_loads_from_https_file(mock_urlopen):

    # codebook data to pass to the mock
    _return_value = json.dumps(
        [
            {
                Features.CODEWORD: [
                    {Indices.ROUND.value: 0, Indices.CH.value: 0, Features.CODE_VALUE: 1},
                    {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1}
                ],
                Features.TARGET: "SCUBE2"
            }
        ]
    ).encode()

    # mock urlopen.read() to return data corresponding to a codebook
    a = MagicMock()
    a.read.side_effect = [_return_value]
    a.__enter__.return_value = a
    mock_urlopen.return_value = a

    # test that the function loads the codebook from the link when called
    codebook = Codebook.from_json('https://www.alink.com/file.json', n_ch=2, n_round=2)
    assert codebook.sizes[Indices.CH] == 2
    assert codebook.sizes[Indices.ROUND] == 2
    assert codebook.sizes[Features.TARGET] == 1
    assert mock_urlopen.call_count == 1
