import os
import sys
import unittest

from ._base_cli_test import CLITest


@unittest.skip("skipping for now")
class TestAllenData(CLITest, unittest.TestCase):

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
                "https://d2nhj9g34unfro.cloudfront.net/20180828/"
                + "allen_smFISH-TEST/allen_smFISH_test_data.zip",
                lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered")
            ],
            [
                "starfish", "Filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "registered/fov_001", "primary_images.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "clip_filtered.json"),
                "Clip",
                "--p-min", "10",
                "--p-max", "100"
            ],
            [
                "starfish", "Filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "clip_filtered.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "bandpass_filtered.json"),
                "Bandpass",
                "--lshort", ".5",
                "--llong", "7",
                "--truncate", "4"
            ],
            [
                "starfish", "Filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "bandpass_filtered.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "clip2_filtered.json"),
                "Clip",
                "--p-min", "10",
                "--p-max", "100"
            ],
            [
                "starfish", "Filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "clip2_filtered.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "gaussian_filtered.json"),
                "GaussianLowPass",
                "--sigma", "1"
            ],
            [
                "starfish", "DetectSpots",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "gaussian_filtered.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results"),
                "LocalMaxPeakFinder",
                "--spot-diameter", "3",
                "--min-mass", "300",
                "--max-size", "3",
                "--separation", "5",
                "--percentile", "10",
                "--is-volume"
            ],

        )

    def verify_results(self, intensities):
        # TODO DEEP SAYS WAIT ON THIS TEST TILL STUFF GETS FIGURED OUT
        pass
