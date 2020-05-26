"""
.. _format_tilefetcher:

Formatting with :py:class:`.TileFetcher`
========================================

The starfish package contains convenient functions for writing SpaceTx formatted experiments with
the :py:mod:`starfish.experiment.builder` module. For data that isn't or can't be structured as
"structured data" as defined in :ref:`format_structured_data`, you can use
:py:func:`.write_experiment_json`. This method is generally only recommended for users familiar
with Python, but we provide many :ref:`examples <data_conversion_examples>` that may allow anyone
to learn and use this workflow for their own data. It requires subclassing the base classes
:py:class:`.TileFetcher` and :py:class:`.FetchedTile` to serve as the interface for fetching the
appropriate 2D image tile. :py:class:`.FetchedTile` can be extended to read any image format that
returns a numpy array.

This tutorial will demonstrate how to define the necessary subclasses and run
:py:func:`.write_experiment_json` on example 2D tiff data.

"""
###################################################################################################
# Create some synthetic data to form into a trivial experiment.
# -------------------------------------------------------------
# The data used here is the same "structured data" as from the :ref:`format_structured_data`
# tutorial. It consists of 2 image sets ('primary' and 'nuclei'), each with 2 FOVs, 2 rounds,
# 1 channel, and 3 z-planes.
#
# The physical coordinates are hardcoded here for simplicity, but typically they will be read
# from a file. Of course, your filename schema and file organization does not have to match
# what is shown here. By writing your own :py:class:`.TileFetcher` subclass you can use any input
# format you want.

import os
import numpy as np
import skimage.io
import tempfile

# columns: r, ch, zplane
fovs = [
    [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
    ],
    [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
    ],
]

data = np.zeros((10, 10), dtype=np.float32)

# create example image tiles that adhere to the structured data schema
inputdir = tempfile.TemporaryDirectory()
primary_dir = os.path.join(inputdir.name, "primary_dir")
nuclei_dir = os.path.join(inputdir.name, "nuclei_dir")
os.mkdir(primary_dir)
os.mkdir(nuclei_dir)

for fov_id, fov in enumerate(fovs):
    for round_label, ch_label, zplane_label in fov:
        primary_path = os.path.join(
            primary_dir, f"primary-f{fov_id}-r{round_label}-c{ch_label}-z{zplane_label}.tiff")
        nuclei_path = os.path.join(
            nuclei_dir, f"nuclei-f{fov_id}-r{round_label}-c{ch_label}-z{zplane_label}.tiff")
        skimage.io.imsave(primary_path, data)
        skimage.io.imsave(nuclei_path, data)

###################################################################################################
# Directory contents
# ------------------

for dir in [primary_dir, nuclei_dir]:
    print("\n")
    for file in sorted(os.listdir(dir)):
        print(file)

###################################################################################################
# Define FetchedTile subclass
# ---------------------------
# The :py:class:`.FetchedTile` subclass defines the function you need for reading your images and
# the other properties required by :py:func:`.write_experiment_json` to construct
# :class:`slicedimage.Tile`\s.
#
# You can use any function to read images that returns a numpy array. We recommend using
# :py:class:`imageio.volread` for 3D images and :py:class:`skimage.io.imread` for 2D images. For
# 3D images, it is especially useful to use a cached function to avoid reopening the file to fetch
# each z-plane.

import functools
from imageio import volread
from skimage.io import imread
from typing import Mapping, Union

from starfish.experiment.builder import FetchedTile
from starfish.types import Axes, Coordinates


# a 2D read function
def read_fn(file_path) -> np.ndarray:
    return imread(file_path)


# example of a cached 3D read function
# not used in this example
@functools.lru_cache(maxsize=1)
def cached_3D_read_fn(file_path) -> np.ndarray:
    return volread(file_path)


# subclass FetchedTile
class RNATile(FetchedTile):

    def __init__(
            self,
            file_path: str,
            coordinates: Mapping[Union[str, Coordinates], tuple]
    ) -> None:
        """Parser for a tile.

        Parameters
        ----------
        file_path : str
            location of the tiff
        coordinates : Mapping[Union[str, Coordinates], tuple]
            the coordinates for the selected tile, extracted from the metadata
        """
        self.file_path = file_path

        # coordinates must match shape
        self._coordinates = coordinates

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: 10, Axes.X: 10}  # hard coded for this example

    @property
    def coordinates(self):
        return self._coordinates

    def tile_data(self) -> np.ndarray:
        return read_fn(self.file_path)


###################################################################################################
# Define TileFetcher subclass
# ---------------------------
# The :py:class:`.TileFetcher` subclass acts as the interface for :py:func:`.write_experiment_json`
# to know where to get files to construct :class:`slicedimage.Tile`\s.
#
# If you are not using structured data, you will need to extend :py:class:`TileFetcher` to reflect
# your own file naming and organization schema.

# physical coordinates for two FOVs
coordinates_of_fovs = [
    {
        Coordinates.X: (0.0, 0.1),
        Coordinates.Y: (0.0, 0.1),
        Coordinates.Z: (0.005, 0.010),
    },
    {
        Coordinates.X: (0.1, 0.2),
        Coordinates.Y: (0.0, 0.1),
        Coordinates.Z: (0.005, 0.010),
    },
]

from starfish.experiment.builder import TileFetcher

class PrimaryTileFetcher(TileFetcher):

    def __init__(self, input_dir: str) -> None:
        self.input_dir = os.path.join(input_dir)
        self.num_z = 1

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        filename = f"primary-f{fov_id}-r{round_label}-c{ch_label}-z{zplane_label}.tiff"
        return RNATile(os.path.join(self.input_dir, filename), coordinates_of_fovs[fov_id])

class NucleiTileFetcher(TileFetcher):

    def __init__(self, input_dir: str) -> None:
        self.input_dir = os.path.join(input_dir)
        self.num_z = 1

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        filename = f"nuclei-f{fov_id}-r{round_label}-c{ch_label}-z{zplane_label}.tiff"
        return RNATile(os.path.join(self.input_dir, filename), coordinates_of_fovs[fov_id])

###################################################################################################
# Run :py:func:`.write_experiment_json`
# -------------------------------------
# The SpaceTx formatted data will all be stored in the output directory you choose.

from slicedimage import ImageFormat
from starfish.experiment.builder import write_experiment_json

outputdir = tempfile.TemporaryDirectory()

primary_tile_fetcher = PrimaryTileFetcher(primary_dir)
nuclei_tile_fetcher = NucleiTileFetcher(nuclei_dir)

# This is hardcoded for this example data set
primary_image_dimensions: Mapping[Union[str, Axes], int] = {
    Axes.ROUND: 2,
    Axes.CH: 1,
    Axes.ZPLANE: 3,
}
aux_images_dimensions: Mapping[str, Mapping[Union[str, Axes], int]] = {
    "nuclei": {
        Axes.ROUND: 2,
        Axes.CH: 1,
        Axes.ZPLANE: 3,
    },
}

write_experiment_json(
    path=outputdir.name,
    fov_count=2,
    tile_format=ImageFormat.TIFF,
    primary_image_dimensions=primary_image_dimensions,
    aux_name_to_dimensions=aux_images_dimensions,
    primary_tile_fetcher=primary_tile_fetcher,
    aux_tile_fetcher={"nuclei": nuclei_tile_fetcher},
    dimension_order=(Axes.ROUND, Axes.CH, Axes.ZPLANE)
)

###################################################################################################
# Don't forget to replace the fake codebook.json
# ----------------------------------------------
# There are no starfish tools for creating a codebook. You can write the JSON manually or write a
# script to do it for you. Be sure the format matches the examples in
# :ref:`SpaceTx Format<sptx_codebook_format>`.

# this is the placeholder codebook.json
with open(os.path.join(outputdir.name, "codebook.json"), "r") as fh:
    print(fh.read())

###################################################################################################
# Load up the experiment
# ----------------------

from starfish import Experiment

exp = Experiment.from_json(os.path.join(outputdir.name, "experiment.json"))
print(exp.fovs())