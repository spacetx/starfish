import difflib
import subprocess
import unittest

from starfish.starfish import art_string
from starfish.util import exec


def assert_diff(actual, expected):
    if isinstance(actual, bytes):
        actual = actual.decode("utf-8")
    if actual != expected:
        diff = difflib.ndiff(expected.splitlines(keepends=True),
                             actual.splitlines(keepends=True))
        diff = list(diff)
        for line in diff:
            print(line, end="")
        raise Exception("help mismatch!")


class TestHelpReturnCode(unittest.TestCase):
    """
    Tests that the CLI supports a '--help' option at all of the expected levels.
    """

    STAGES = (
        [
            "starfish", "--help",
            lambda tempdir: tempdir
        ],
        [
            "starfish", "detect_spots", "--help",
            lambda tempdir: tempdir
        ],
        [
            "starfish", "detect_spots", "BlobDetector", "--help",
            lambda tempdir: tempdir
        ],
    )

    def test_run_build(self):
        exec.stages(
            TestHelpReturnCode.STAGES,
            keep_data=False)

class TestHelpStandardOut(unittest.TestCase):
    """
    Tests that the calls to CLI's help produce the output that users expect.
    """

    def test_first(self):
        actual = subprocess.check_output(["starfish", "--help"])
        expected = """Usage: starfish [OPTIONS] COMMAND [ARGS]...

Options:
  --profile
  --help     Show this message and exit.

Commands:
  build
  decode
  detect_spots
  filter
  registration
  segment
  target_assignment
  validate           invokes validate with the parsed commandline...
  version
"""
        assert_diff(actual, expected)

    def test_second(self):
        actual = subprocess.check_output(["starfish", "detect_spots", "--help"])
        expected = """%s
Usage: starfish detect_spots [OPTIONS] COMMAND [ARGS]...

Options:
  -i, --input PATH                [required]
  -o, --output TEXT               [required]
  --blobs-stack TEXT              ImageStack that contains the blobs. Will be
                                  max-projected across imaging round and
                                  channel to produce the blobs_image
  --reference-image-from-max-projection
                                  Construct a reference image by max
                                  projecting imaging rounds and channels.
                                  Spots are found in this image and then
                                  measured across all images in the input
                                  stack.
  --codebook TEXT                 A spaceTx spec-compliant json file that
                                  describes a three dimensional tensor whose
                                  values are the expected intensity of a spot
                                  for each code in each imaging round and each
                                  color channel.
  --help                          Show this message and exit.

Commands:
  BlobDetector
  PixelSpotDetector
  TrackpyLocalMaxPeakFinder
""" % art_string()
        actual = actual.decode("utf-8")
        assert_diff(actual, expected)

    def test_third(self):
        actual = subprocess.check_output(["starfish", "detect_spots", "BlobDetector", "--help"])
        expected = """Usage: starfish detect_spots BlobDetector [OPTIONS]

Options:
  --min-sigma INTEGER     Minimum spot size (in standard deviation)
  --max-sigma INTEGER     Maximum spot size (in standard deviation)
  --num-sigma INTEGER     Number of sigmas to try
  --threshold FLOAT       Dots threshold
  --overlap FLOAT         dots with overlap of greater than this fraction are
                          combined
  --show                  display results visually
  --detector_method TEXT  str ['blob_dog', 'blob_doh', 'blob_log'] name of the
                          type of detection method used from skimage.feature
  --help                  Show this message and exit.
"""
        assert_diff(actual, expected)
