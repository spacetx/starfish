"""
.. _tilefetcher_loader:

Loading Data through TileFetchers
=================================

:py:class:`.TileFetcher`\s provide a way for starfish's data
formatting tooling to obtain image data and metadata for each tile that make up an image.  It is
possible to use this interface to directly load data into starfish.  This could have performance
implications as the TileFetcher's performance could potentially be highly sensitive to the order
the tiles are retrieved.  Consider an example where the data for a 5D dataset is stored as a
series of X-Y-Z-C 4D files (i.e., one file per round).  If we load all the data for a round
consecutively (i.e., hold round constant as long as possible), we will have better performance than
we would if we load all the data for a channel consecutively (i.e., round will vary rapidly, but
channel will be held constant as long as possible).  The former will open each 4D file once whereas
the latter would open each file and read a single tile repeatedly.  However, this approach may be
suitable if conversion is deemed too costly.

A demonstration of this functionality will be produced using synthetic data.
"""

###################################################################################################
# Create some synthetic 5D data
# -----------------------------
from typing import Mapping, Union

import numpy as np

from starfish.core.types import Coordinates, CoordinateValue, Axes

tile_2d_shape = (120, 30)
num_z = 5
num_r = 4
num_c = 3

synthetic_data = np.random.random(size=(num_r, num_c, num_z) + tile_2d_shape).astype(np.float32)

###################################################################################################
# Write as a series of 3D tiffs.
# -------------------------------------------------------------

import os, tempfile
from imageio import volread, volwrite

dir = tempfile.TemporaryDirectory()

for r in range(num_r):
    for c in range(num_c):
        volwrite(os.path.join(dir.name, f"r{r}_c{c}.tiff"), synthetic_data[r, c])

###################################################################################################
# Now build a FetchedTile and TileFetcher based on this data.
# -----------------------------------------------------------

import functools
from starfish.experiment.builder import FetchedTile, TileFetcher

# We use this to cache images across tiles.  To avoid reopening and decoding the TIFF file, we use a
# single-element cache that maps between file_path and the array data.
@functools.lru_cache(maxsize=1)
def cached_read_fn(file_path) -> np.ndarray:
    return volread(file_path, format="tiff")

class DemoFetchedTile(FetchedTile):
    def __init__(self, filename, z, *args, **kwargs):
        self.filename = filename
        self.z = z

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {
            Axes.Y: tile_2d_shape[0], Axes.X: tile_2d_shape[1],
        }

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return {
            Coordinates.X: (0, 0.001),
            Coordinates.Y: (0, 0.001),
            Coordinates.Z: (0.001 * self.z, 0.001 * (self.z + 1)),
        }

    def tile_data(self) -> np.ndarray:
        return cached_read_fn(self.filename)[self.z]

class DemoTileFetcher(TileFetcher):
    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        return DemoFetchedTile(os.path.join(dir.name, f"r{r}_c{c}.tiff"), zplane_label)

###################################################################################################
# Load the data as an ImageStack.
# -------------------------------

from starfish import ImageStack

stack = ImageStack.from_tilefetcher(
    DemoTileFetcher(),
    {
        Axes.Y: tile_2d_shape[0], Axes.X: tile_2d_shape[1],
    },
    fov=0,
    rounds=range(num_r),
    chs=range(num_c),
    zplanes=range(num_z),
    group_by=(Axes.ROUND, Axes.CH),
)
print(repr(stack))
