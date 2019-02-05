import os
import unittest

from starfish.util import exec


class TestWithBuildData(unittest.TestCase):

    STAGES = (
        [
            "starfish", "build",
            "--fov-count=2", '--primary-image-dimensions={"z": 3}',
            lambda tempdir: tempdir
        ],
        # Old-style
        [
            "starfish", "validate", "--experiment-json",
            lambda tempdir: os.sep.join([tempdir, "experiment.json"])
        ],
        # New-style
        [
            "starfish", "validate", "experiment",
            lambda tempdir: os.sep.join([tempdir, "experiment.json"])
        ],
    )

    def test_run_build(self):
        exec.stages(
            TestWithBuildData.STAGES,
            keep_data=("TEST_BUILD_KEEP_DATA" in os.environ))
