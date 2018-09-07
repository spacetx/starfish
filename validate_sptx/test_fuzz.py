import os
from pkg_resources import resource_filename

from .util import SpaceTxValidator
from .util import Fuzzer

codebook_schema_path = resource_filename("validate_sptx", "schema/codebook/codebook.json")
experiment_schema_path = resource_filename("validate_sptx", "schema/experiment.json")
codebook_validator = SpaceTxValidator(codebook_schema_path)
experiment_validator = SpaceTxValidator(experiment_schema_path)


def test_fuzz_mock():
    obj = {
        "list": [1, 2, 3]
    }
    values = [
        { "list": [1, 2, 3], "fake": "!" },
        { },
        { "list": 123456789 },
        { "list": "fake" },
        { "list": {} },
        { "list": [] },
        { "list": [1, 2, 3, '!'] },
        { "list": [2, 3] },
        { "list": [123456789, 2, 3] },
        { "list": ["fake", 2, 3] },
        { "list": [{}, 2, 3] },
        { "list": [[], 2, 3] },
        { "list": [1, 2, 3, "!"]},
        { "list": [1, 3] },
        { "list": [1, 123456789, 3] },
        { "list": [1, "fake", 3] },
        { "list": [1, {}, 3] },
        { "list": [1, [], 3] },
        { "list": [1, 2, 3, '!'] },
        { "list": [1, 2] },
        { "list": [1, 2, 123456789] },
        { "list": [1, 2, "fake"] },
        { "list": [1, 2, {}] },
        { "list": [1, 2, []] },
    ]
    class Validator(object):
        def __init__(self):
            self.called = 0
        def is_valid(self, obj):
            self.called += 1
            assert obj == values.pop(0), self.called
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
    assert codebook_validator.validate_object(codebook, fuzz=True)


def test_fuzz_experiment():
    experiment = {
        "version": "0.0.0",
        "primary_images": "primary_images.json",
        "auxiliary_images": {
          "nuclei": "nuclei.json"
        },
        "codebook": "codebook.json",
        "extras": {
          "is_space_tx_cool": True
        }
      }
    assert experiment_validator.validate_object(experiment, fuzz=True)
