import os
import unittest

from starfish.core.util import exec


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
        # Validate other input files
        [
            "starfish", "validate", "experiment",
            lambda tempdir, *args, **kwargs: os.sep.join([tempdir, "experiment.json"])
        ],
        [
            "starfish", "validate", "codebook",
            lambda tempdir, *args, **kwargs: os.sep.join([tempdir, "codebook.json"])
        ],
        [
            "starfish", "validate", "manifest",
            lambda tempdir, *args, **kwargs: os.sep.join([tempdir, "primary_images.json"])
        ],
        [
            "starfish", "validate", "fov",
            lambda tempdir, *args, **kwargs: os.sep.join([tempdir, "primary_images-fov_000.json"])
        ],
    )

    def test_run_build(self):
        exec.stages(
            TestWithBuildData.STAGES,
            keep_data=("TEST_BUILD_KEEP_DATA" in os.environ))
