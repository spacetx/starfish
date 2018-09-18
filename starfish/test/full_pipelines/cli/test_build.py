import os
import unittest

from starfish.util import exec


class TestWithBuildData(unittest.TestCase):

    STAGES = (
        [
            "starfish", "build",
            "--fov-count=2", '--hybridization-dimensions={"z": 3}',
            lambda tempdir: tempdir
        ],
        [
            "starfish", "validate", "--experiment-json",
            lambda tempdir: os.sep.join([tempdir, "experiment.json"])
        ],
    )

    def test_run_build(self):
        exec.stages(
            TestWithBuildData.STAGES,
            keep_data=("TEST_BUILD_KEEP_DATA" in os.environ))
