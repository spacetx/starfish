import os
import sys
import unittest

from ._base_cli_test import CLITest


@unittest.skip("skipping for now")
class TestWithMerfishData(CLITest, unittest.TestCase):
    # __test__ = True
    # TODO this test currently fails because it doesn't do the scaling the notebook does in memory
    # Need to figure out ScaleByPercentile change

    @property
    def subdirs(self):
        return (
            "registered",
            "filtered",
            "results"
        )

    @property
    def stages(self):
        return (
            [
                sys.executable,
                "starfish/test/full_pipelines/cli/get_cli_test_data.py",
                "https://d2nhj9g34unfro.cloudfront.net/20181005/MERFISH-TEST/experiment.json",
                lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered")
            ],
            [
                "starfish", "Filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "registered/fov_001", "primary_images.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "gaussian_filtered.json"),
                "GaussianHighPass",
                "--sigma", "3",
            ],
            [
                "starfish", "Filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "gaussian_filtered.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "deconvolve_filtered.json"),
                "DeconvolvePSF",
                "--sigma", "2",
                "--num-iter", "9"
            ],
            [
                "starfish", "Filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "deconvolve_filtered.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "gaussian_filtered.json"),
                "GaussianLowPass",
                "--sigma", "1"
            ],
            [
                "starfish", "Filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "deconvolve_filtered.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "scale_filtered.json"),
                "ScaleByPercentile",
                "--p", "90",
            ],
            [
                "starfish", "DetectSpots",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "scale_filtered.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results"),
                "PixelSpotDetector",
                "--codebook", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "registered", "codebook.json"),
                "--distance-threshold", "0.5176",
                "--magnitude-threshold", "5e-5",
                "--norm-order", "2",
            ],
        )

    def verify_results(self, intensities):
        pass
