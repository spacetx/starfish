import os

from starfish.types import Axes, Features
from .factories import simple_codebook_array
from ..codebook import Codebook


def test_to_json(tmp_path):
    code_array = simple_codebook_array()
    codebook = Codebook.from_code_array(code_array)
    codebook_path = tmp_path / "codebook.json"
    codebook.to_json(codebook_path)

    loaded_codebook = Codebook.open_json(os.fspath(codebook_path))
    assert codebook.equals(loaded_codebook)


def test_to_json_multiple_codes_for_target(tmp_path):
    code_array = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "SCUBE2"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "BRCA"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "SCUBE2"
        }
    ]
    codebook = Codebook.from_code_array(code_array)
    codebook_path = tmp_path / "codebook.json"
    codebook.to_json(codebook_path)

    loaded_codebook = Codebook.open_json(os.fspath(codebook_path))
    assert codebook.equals(loaded_codebook)
