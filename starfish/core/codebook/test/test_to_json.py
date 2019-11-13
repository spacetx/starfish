import os

from .factories import simple_codebook_array
from ..codebook import Codebook


def test_to_json(tmp_path):
    code_array = simple_codebook_array()
    codebook = Codebook.from_code_array(code_array)
    codebook_path = tmp_path / "codebook.json"
    codebook.to_json(codebook_path)

    loaded_codebook = Codebook.open_json(os.fspath(codebook_path))
    assert codebook.equals(loaded_codebook)
