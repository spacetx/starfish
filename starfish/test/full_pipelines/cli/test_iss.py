"""
Notes
-----
This test and docs/source/usage/iss/iss_cli.sh test the same code paths and should be updated
together
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

from starfish.test.full_pipelines.cli._base_cli_test import CLITest
from starfish.types import Features


class TestWithIssData(CLITest, unittest.TestCase):

    @property
    def spots_file(self):
        return "decoded-spots.nc"

    @property
    def subdirs(self):
        return (
            "raw",
            "formatted",
            "registered",
            "filtered",
            "results",
        )

    @property
    def stages(self):
        return (
            [
                sys.executable,
                "examples/get_cli_test_data.py",
                "https://d2nhj9g34unfro.cloudfront.net/20181005/ISS-TEST/",
                lambda tempdir, *args, **kwargs: os.path.join(tempdir, "formatted")
            ],
            [
                "starfish", "registration",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "formatted/fov_001", "hybridization.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "registered", "hybridization.json"),
                "FourierShiftRegistration",
                "--reference-stack", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "formatted/fov_001", "dots.json"),
                "--upsampling", "1000",
            ],
            [
                "starfish", "filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "registered", "hybridization.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "hybridization.json"),
                "WhiteTophat",
                "--masking-radius", "15",
            ],
            [
                "starfish", "filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "formatted/fov_001", "nuclei.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "nuclei.json"),
                "WhiteTophat",
                "--masking-radius", "15",
            ],
            [
                "starfish", "filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "formatted/fov_001", "dots.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "dots.json"),
                "WhiteTophat",
                "--masking-radius", "15",
            ],
            [
                "starfish", "detect_spots",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "hybridization.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "spots.nc"),
                "--blobs-stack", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "dots.json"),
                "GaussianSpotDetector",
                "--min-sigma", "4",
                "--max-sigma", "6",
                "--num-sigma", "20",
                "--threshold", "0.01",
            ],
            [
                "starfish", "segment",
                "--hybridization-stack", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "hybridization.json"),
                "--nuclei-stack", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "nuclei.json"),
                "-o", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "regions.geojson"),
                "Watershed",
                "--dapi-threshold", ".16",
                "--input-threshold", ".22",
                "--min-distance", "57",
            ],
            [
                "starfish", "target_assignment",
                "--coordinates-geojson",
                lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "regions.geojson"),
                "--intensities", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "spots.nc"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "targeted-spots.nc"),
                "PointInPoly2D",
            ],
            [
                "starfish", "decode",
                "-i", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "targeted-spots.nc"),
                "--codebook", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "formatted", "codebook.json"),
                "-o", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "decoded-spots.nc"),
                "PerRoundMaxChannelDecoder",
            ],
        )

    def verify_results(self, intensities):
        # TODO make this test stronger
        genes, counts = np.unique(
            intensities.coords[Features.TARGET], return_counts=True)
        gene_counts = pd.Series(counts, genes)
        # TODO THERE"S NO HUMAN/MOUSE KEYS?
        assert gene_counts['ACTB']
