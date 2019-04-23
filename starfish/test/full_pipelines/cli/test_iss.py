"""
Notes
-----
This test and docs/source/usage/iss/iss_cli.sh test the same code paths and should be updated
together
"""
import os
import unittest

import numpy as np
import pandas as pd
import pytest

from starfish.types import Features
from ._base_cli_test import CLITest


EXPERIMENT_JSON_URL = "https://d2nhj9g34unfro.cloudfront.net/20181005/ISS-TEST/experiment.json"


@pytest.mark.slow
class TestWithIssData(CLITest, unittest.TestCase):

    @property
    def spots_file(self):
        return "decoded-spots.nc"

    @property
    def subdirs(self):
        return (
            "max_projected",
            "transforms",
            "registered",
            "filtered",
            "results",
        )

    @property
    def stages(self):
        return (
            [
                "starfish", "validate", "experiment", EXPERIMENT_JSON_URL,
            ],
            [
                "starfish", "Filter",
                "--input",
                f"@{EXPERIMENT_JSON_URL}[fov_001][primary]",
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "max_projected", "primary_images.json"),
                "MaxProject",
                "--dims", "c",
                "--dims", "z"

            ],
            [
                "starfish", "LearnTransform",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "max_projected", "primary_images.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "transforms", "transforms.json"),
                "Translation",
                "--reference-stack",
                f"@{EXPERIMENT_JSON_URL}[fov_001][dots]",
                "--upsampling", "1000",
                "--axes", "r"
            ],
            [
                "starfish", "ApplyTransform",
                "--input",
                f"@{EXPERIMENT_JSON_URL}[fov_001][primary]",
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "registered", "primary_images.json"),
                "--transformation-list", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "transforms", "transforms.json"),
                "Warp",
            ],
            [
                "starfish", "Filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "registered", "primary_images.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "primary_images.json"),
                "WhiteTophat",
                "--masking-radius", "15",
            ],
            [
                "starfish", "Filter",
                "--input",
                f"@{EXPERIMENT_JSON_URL}[fov_001][nuclei]",
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "nuclei.json"),
                "WhiteTophat",
                "--masking-radius", "15",
            ],
            [
                "starfish", "Filter",
                "--input",
                f"@{EXPERIMENT_JSON_URL}[fov_001][dots]",
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "dots.json"),
                "WhiteTophat",
                "--masking-radius", "15",
            ],
            [
                "starfish", "DetectSpots",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "primary_images.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "spots.nc"),
                "--blobs-stack", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "dots.json"),
                "--blobs-axis", "r", "--blobs-axis", "c",
                "BlobDetector",
                "--min-sigma", "4",
                "--max-sigma", "6",
                "--num-sigma", "20",
                "--threshold", "0.01",
            ],
            [
                "starfish", "Segment",
                "--primary-images", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "primary_images.json"),
                "--nuclei", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "nuclei.json"),
                "-o", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "masks.tgz"),
                "Watershed",
                "--nuclei-threshold", ".16",
                "--input-threshold", ".22",
                "--min-distance", "57",
            ],
            [
                "starfish", "AssignTargets",
                "--label-image",
                lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "masks.tgz"),
                "--intensities", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "spots.nc"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "targeted-spots.nc"),
                "Label",
            ],
            [
                "starfish", "Decode",
                "-i", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "targeted-spots.nc"),
                "--codebook",
                f"@{EXPERIMENT_JSON_URL}",
                "-o", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "decoded-spots.nc"),
                "PerRoundMaxChannel",
            ],

            # Validate results/{spots,targeted-spots,decoded-spots}.nc
            [
                "starfish", "validate", "xarray",
                lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "spots.nc")
            ],
            [
                "starfish", "validate", "xarray",
                lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "targeted-spots.nc")
            ],
            [
                "starfish", "validate", "xarray",
                lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "decoded-spots.nc")
            ],
        )

    def verify_results(self, intensities):
        # TODO make this test stronger
        genes, counts = np.unique(
            intensities.coords[Features.TARGET], return_counts=True)
        gene_counts = pd.Series(counts, genes)
        # TODO THERE"S NO HUMAN/MOUSE KEYS?
        assert gene_counts['ACTB']
