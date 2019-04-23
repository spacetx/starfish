"""
Notes
-----
This test and docs/source/usage/iss/iss_cli.sh test the same code paths and should be updated
together
"""
import os
import unittest
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from starfish import IntensityTable
from starfish.types import Features
from ._base_recipe_test import RecipeTest


URL = "https://d2nhj9g34unfro.cloudfront.net/20181005/ISS-TEST/experiment.json"


@pytest.mark.slow
class TestWithIssData(RecipeTest, unittest.TestCase):
    @property
    def recipe(self) -> Path:
        test_file_path = Path(__file__)
        recipe = test_file_path.parent / "iss_recipe.txt"
        return recipe

    @property
    def input_url_or_paths(self) -> Iterable[str]:
        return [
            f"@{URL}[fov_001][primary]",  # primary image
            f"@{URL}[fov_001][dots]",     # dots image
            f"@{URL}[fov_001][nuclei]",   # nuclei image
            f"@{URL}",                    # codebook
        ]

    @property
    def output_paths(self) -> Iterable[Path]:
        return [
            Path("decoded_spots.nc")
        ]

    def verify_results(self, tempdir: Path):
        intensities = IntensityTable.open_netcdf(os.fspath(tempdir / "decoded_spots.nc"))
        genes, counts = np.unique(
            intensities.coords[Features.TARGET], return_counts=True)
        gene_counts = pd.Series(counts, genes)
        assert gene_counts['ACTB'] == 9
        assert gene_counts['GAPDH'] == 9
