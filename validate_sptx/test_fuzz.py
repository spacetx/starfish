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
        { "list": [1, 2, 3, '!'] },
        { "list": [2, 3] },
        { "list": [123456789, 2, 3] },
        { "list": ["fake", 2, 3] },
        { "list": [1, 2, 3, '!'] },
        { "list": [1, 3] },
        { "list": [1, 123456789, 3] },
        { "list": [1, "fake", 3] },
        { "list": [1, 2, 3, '!'] },
        { "list": [1, 2] },
        { "list": [1, 2, 123456789] },
        { "list": [1, 2, "fake"] },
    ]
    class Validator(object):
        def is_valid(self, obj):
            assert obj == values.pop(0)
            return True
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
