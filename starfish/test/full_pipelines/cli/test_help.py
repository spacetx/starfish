import subprocess
import unittest

from starfish.util import exec


def assert_contains(actual, expected):
    if isinstance(actual, bytes):
        actual = actual.decode("utf-8")
    if expected not in actual:
        raise Exception(f"counldn't find: 'f{expected}'")


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
        expected = """Usage: starfish [OPTIONS] COMMAND [ARGS]..."""
        assert_contains(actual, expected)

    def test_second(self):
        actual = subprocess.check_output(["starfish", "detect_spots", "--help"])
        expected = """Usage: starfish detect_spots [OPTIONS] COMMAND [ARGS]..."""
        actual = actual.decode("utf-8")
        assert_contains(actual, expected)

    def test_third(self):
        actual = subprocess.check_output(["starfish", "detect_spots", "BlobDetector", "--help"])
        expected = """Usage: starfish detect_spots BlobDetector [OPTIONS]"""
        assert_contains(actual, expected)
