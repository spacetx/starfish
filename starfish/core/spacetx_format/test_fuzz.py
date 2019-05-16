import io

from starfish.core.codebook._format import CURRENT_VERSION, DocumentKeys
from .util import Fuzzer, LatestCodebookValidator, LatestExperimentValidator

package_name = "starfish"
codebook_validator = LatestCodebookValidator()
experiment_validator = LatestExperimentValidator()


def test_fuzz_mock():
    """
    Starting from the simple obj test class, the Fuzzer will
    propose all the different mutations contained in the the
    values list. The mocked validator will pop them off, in
    order, and compare what it is being given. For testing
    purposes, it will always return true.
    """
    obj = {
        "list": [1, 2, 3]
    }
    values = [
        {"list": [1, 2, 3], "fake": "!"},
        {},
        {"list": 123456789},
        {"list": "fake"},
        {"list": {}},
        {"list": []},
        {"list": [1, 2, 3, '!']},
        {"list": [2, 3]},
        {"list": [123456789, 2, 3]},
        {"list": ["fake", 2, 3]},
        {"list": [{}, 2, 3]},
        {"list": [[], 2, 3]},
        {"list": [1, 2, 3, "!"]},
        {"list": [1, 3]},
        {"list": [1, 123456789, 3]},
        {"list": [1, "fake", 3]},
        {"list": [1, {}, 3]},
        {"list": [1, [], 3]},
        {"list": [1, 2, 3, '!']},
        {"list": [1, 2]},
        {"list": [1, 2, 123456789]},
        {"list": [1, 2, "fake"]},
        {"list": [1, 2, {}]},
        {"list": [1, 2, []]},
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
    """
    simple validation of a hard-coded example codebook.

    The actual values don't matter overly much, but it
    provides a good example of the output.
    """
    codebook = {
        DocumentKeys.VERSION_KEY: str(CURRENT_VERSION),
        DocumentKeys.MAPPINGS_KEY: [
            {
                "codeword": [
                    {"r": 0, "c": 0, "v": 1},
                    {"r": 0, "c": 1, "v": 1},
                    {"r": 1, "c": 1, "v": 1}
                ],
                "target": "SCUBE2"
            }
        ]
    }
    out = io.StringIO()
    codebook_validator.fuzz_object(codebook, out=out)
    expect = """> Fuzzing unknown...
A D I S M L	If the letter is present, mutation is valid!
-----------	--------------------------------------------
. . . . . .	version:
. . . . . .	   0.0.0
. . . . . .	mappings:
A . . . . .	  - codeword:
A D I . . .	    - r:
A D I . . .	         0
A . I . . .	      c:
A . I . . .	         0
A D . . . .	      v:
A D . . . .	         1
A D I . . .	    - r:
A D I . . .	         0
A . I . . .	      c:
A . I . . .	         1
A D . . . .	      v:
A D . . . .	         1
A D I . . .	    - r:
A D I . . .	         1
A . I . . .	      c:
A . I . . .	         1
A D . . . .	      v:
A D . . . .	         1
A . . S . .	    target:
A . . S . .	       SCUBE2
"""
    assert expect == out.getvalue()


def test_fuzz_experiment():
    """
    simple validation of a hard-coded example experiment.

    The actual values don't matter overly much, but it
    provides a good example of the output.
    """
    experiment = {
        "version": "0.0.0",
        "images": {
            "primary": "primary_images.json",
            "nuclei": "nuclei.json",
        },
        "codebook": "codebook.json",
        "extras": {
            "is_space_tx_cool": True
        }
    }
    out = io.StringIO()
    experiment_validator.fuzz_object(experiment, out=out)
    expect = """> Fuzzing unknown...
A D I S M L	If the letter is present, mutation is valid!
-----------	--------------------------------------------
. . . . . .	version:
. . . . . .	   0.0.0
. . . . . .	images:
. . . . . .	   primary:
. . . . . .	      primary_images.json
. D . . . .	   nuclei:
. D . . . .	      nuclei.json
. . . . . .	codebook:
. . . . . .	   codebook.json
. D I S M L	extras:
A D I S M L	   is_space_tx_cool:
A D I S M L	      True
"""
    assert expect == out.getvalue()
