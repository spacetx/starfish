import os
import sys

from starfish.types import Features
from starfish.test.full_pipelines.cli._base_cli_test import CLITest


class TestWithDartfishData(CLITest):
    __test__ = True

    SUBDIRS = (
        "registered",
        "filtered",
        "results"
    )

    STAGES = (
        [
            sys.executable,
            "examples/get_cli_test_data.py",
            "https://dmf0bdeheu4zf.cloudfront.net/20180828/DARTFISH-TEST/dartfish-test-data.zip",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered")
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "registered/fov_001", "hybridization.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "hybridization.json"),
            "ScaleByPercentile",
            "--p", "100",
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(
            tempdir, "filtered", "hybridization.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
            tempdir, "filtered", "filtered.json"),
            "ZeroByChannelMagnitude",
            "--thresh", ".05",
            "--normalize", "False"
        ],
        [
            "starfish", "detect_spots",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "filtered.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results"),
            "PixelSpotDetector",
            "--codebook", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "registered", "codebook.json"),
            "--distance-threshold", "3",
            "--magnitude-threshold", ".5",
            "--min-area", "5",
            "--max-area", "30",
        ],
    )

    def verify_results(self, intensities):
        # assert intensities.sizes[Features.AXIS] == 53
        #TODO update once float changes merge
        pass


