"""
Running a Pipeline
==================
This example loads one of the pipeline recipes used in starfish's unit tests and executes it from
the recipe API.  Subsequently, we execute the same recipe using the CLI.
"""

###################################################################################################
# This is a hack to load the recipe from the existing code.
import os
import pathlib
from starfish.test.full_pipelines import recipe

recipe_tests_folder = pathlib.Path(recipe.__file__).parent
iss_recipe_file = recipe_tests_folder / "iss_recipe.txt"

with open(os.fspath(iss_recipe_file), "r") as fh:
    recipe_contents = fh.read()

print(recipe_contents)

###################################################################################################
# Providing data
# --------------
#
# The recipe references `file_input[0]`...`file_input[3]`.  We provide values for these variables to
# the recipe, and when the recipe executes, the `file_input[n]` references are replaced by the
# values.
#
# Recipes support referencing an image or codebook within an experiment through a special syntax:
# ``@<url_or_path>[fov_name][img_name]`` will load the experiment.json from <url_or_path> and return
# the ``img_name`` image form the ``fov_name`` fov.

experiment_url = "https://d2nhj9g34unfro.cloudfront.net/20181005/ISS-TEST/experiment.json"
primary_image = f"@{experiment_url}[fov_001][primary]"
dots_image = f"@{experiment_url}[fov_001][dots]"
nuclei_image = f"@{experiment_url}[fov_001][nuclei]"
codebook = f"@{experiment_url}"
print(
    f"primary_image: {primary_image}\n"
    f"dots_image: {dots_image}\n"
    f"nuclei_image: {nuclei_image}\n"
    f"codebook: {codebook}\n")

###################################################################################################
# Writing out data
# ----------------
#
# Any steps that assign a value to ``file_output`` will require an output file.  Let's set up a
# directory to write our outputs to.

import tempfile
tempdir = tempfile.TemporaryDirectory()
output_path_api = pathlib.Path(tempdir.name) / "decoded_spots_api.nc"
output_path_cli = pathlib.Path(tempdir.name) / "decoded_spots_cli.nc"

###################################################################################################
# Execute the recipe
# ------------------
#
# Let's execute the recipe!

from starfish.core.recipe import Recipe
recipe = Recipe(
    recipe_contents,
    [primary_image, dots_image, nuclei_image, codebook],
    [os.fspath(output_path_api)]
)
recipe.run_and_save()

###################################################################################################
# Load up results
# ---------------
#
# We can now load up the results.
import numpy as np
import pandas as pd
from starfish import IntensityTable
from starfish.types import Features

intensity_table = IntensityTable.open_netcdf(os.fspath(output_path_api))

genes, counts = np.unique(
    intensity_table.coords[Features.TARGET], return_counts=True)
gene_counts = pd.Series(counts, genes)
print(gene_counts)

###################################################################################################
# Execute the recipe (from the command line)
# ------------------------------------------
#
# We can also execute the recipe from the command line.  The command line will look like
cmdline = [
    "starfish",
    "recipe",
    "--recipe",
    os.fspath(iss_recipe_file),
]
for input_file in (primary_image, dots_image, nuclei_image, codebook):
    cmdline.append("--input")
    cmdline.append(input_file)
cmdline.append("--output")
cmdline.append(os.fspath(output_path_cli))

cmdline_str = " ".join(cmdline)

print(f"% {cmdline_str}")

import subprocess
subprocess.check_output(cmdline)

###################################################################################################
# Load up results from the command line invocation
# ------------------------------------------------
#
# This should produce identical results to the API invocation.
intensity_table = IntensityTable.open_netcdf(os.fspath(output_path_cli))

genes, counts = np.unique(
    intensity_table.coords[Features.TARGET], return_counts=True)
gene_counts = pd.Series(counts, genes)
print(gene_counts)

