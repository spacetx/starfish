import os
from pkg_resources import resource_filename

from .util import SpaceTxValidator
from .util import Fuzzer

codebook_schema_path = resource_filename("validate_sptx", "schema/codebook/codebook.json")
validator = SpaceTxValidator(codebook_schema_path)


def test_fuzz_mock():
    obj = {
        "list": [1, 2, 3]
    }
    values = [
        obj,
        obj,
        obj,
    ]
    class Validator(object):
        def is_valid(self, obj):
            assert obj == values.pop(0)
    Fuzzer(Validator(), obj).fuzz()


def test_fuzz_codebook():
    codebook = [
        {
          "codeword": [
            {"r": 0, "c": 0, "v": 1},
            {"r": 0, "c": 1, "v": 1},
            {"r": 1, "c": 1, "v": 1}
          ],
          "target": "SCUBE2"
        }
    ]
    assert validator.validate_object(codebook, fuzz=True)
