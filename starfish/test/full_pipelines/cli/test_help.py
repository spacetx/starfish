import unittest

from starfish.util import exec


class TestHelp(unittest.TestCase):

    STAGES = (
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
            TestHelp.STAGES,
            keep_data=False)
