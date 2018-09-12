import os
import sys

from starfish.test.full_pipelines.cli._base_cli_test import CLITest


class TestWithMerfishData(CLITest):
    # __test__ = True
    # TODO this test currently fails because it doesn't do the scaling the notebook does in memory
    # Need to figure out ScaleByPercentile change

    SUBDIRS = (
        "registered",
        "filtered",
        "results"
    )

    STAGES = (
        [
            sys.executable,
            "examples/get_cli_test_data.py",
            "https://dmf0bdeheu4zf.cloudfront.net/20180828/MERFISH-TEST/merfish-test-data.zip",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered")
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "registered/fov_001", "hybridization.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "gaussian_filtered.json"),
            "GaussianHighPass",
            "--sigma", "3",
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "gaussian_filtered.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "deconvolve_filtered.json"),
            "DeconvolvePSF",
            "--sigma", "2",
            "--num-iter", "9"
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "deconvolve_filtered.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "gaussian_filtered.json"),
            "GaussianLowPass",
            "--sigma", "1"
        ],
        [
            "starfish", "detect_spots",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "gaussian_filtered.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results"),
            "PixelSpotDetector",
            "--codebook", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "registered", "codebook.json"),
            "--distance-threshold", "0.5176",
            "--magnitude-threshold", "5e-5",
            "--norm-order", "2",
            "--crop-x", "0",
            "--crop-y", "40",
            "--crop-z", "40"
        ],
    )

    def verify_results(self, intensities):
        pass
