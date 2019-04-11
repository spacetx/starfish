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

from starfish.test.full_pipelines.cli._base_cli_test import CLITest
from starfish.types import Features


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
                "starfish", "filter",
                "--input",
                f"@{EXPERIMENT_JSON_URL}[fov_001][primary]",
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "max_projected", "primary_images.json"),
                "MaxProj",
                "--dims", "c",
                "--dims", "z"

            ],
            [
                "starfish", "learn_transform",
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
                "starfish", "apply_transform",
                "--input",
                f"@{EXPERIMENT_JSON_URL}[fov_001][primary]",
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "registered", "primary_images.json"),
                "--transformation-list", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "transforms", "transforms.json"),
                "Warp",
            ],
            [
                "starfish", "filter",
                "--input", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "registered", "primary_images.json"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "primary_images.json"),
                "WhiteTophat",
                "--masking-radius", "15",
            ],
            [
                "starfish", "filter",
                "--input",
                f"@{EXPERIMENT_JSON_URL}[fov_001][nuclei]",
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "nuclei.json"),
                "WhiteTophat",
                "--masking-radius", "15",
            ],
            [
                "starfish", "filter",
                "--input",
                f"@{EXPERIMENT_JSON_URL}[fov_001][dots]",
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "dots.json"),
                "WhiteTophat",
                "--masking-radius", "15",
            ],
            [
                "starfish", "detect_spots",
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
                "starfish", "segment",
                "--primary-images", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "primary_images.json"),
                "--nuclei", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "filtered", "nuclei.json"),
                "-o", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "label_image.png"),
                "Watershed",
                "--nuclei-threshold", ".16",
                "--input-threshold", ".22",
                "--min-distance", "57",
            ],
            [
                "starfish", "target_assignment",
                "--label-image",
                lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "label_image.png"),
                "--intensities", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "spots.nc"),
                "--output", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "targeted-spots.nc"),
                "Label",
            ],
            [
                "starfish", "decode",
                "-i", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "targeted-spots.nc"),
                "--codebook",
                f"@{EXPERIMENT_JSON_URL}",
                "-o", lambda tempdir, *args, **kwargs: os.path.join(
                    tempdir, "results", "decoded-spots.nc"),
                "PerRoundMaxChannelDecoder",
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
