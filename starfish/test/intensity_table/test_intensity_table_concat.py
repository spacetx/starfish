import numpy as np
import xarray as xr

from starfish import IntensityTable
from starfish.test import test_utils
from starfish.types import Coordinates


def test_overlap():
    codebook = test_utils.codebook_array_factory()
    it1 = IntensityTable.synthetic_intensities(
        codebook,
        num_z=1,
        height=50,
        width=50,
        n_spots=10
    )
    # intensity table 1 has 10 spots, xmin = 0, ymin = 0, xmax = 2, ymax = 1
    it1[Coordinates.X.value] = xr.DataArray(np.linspace(0, 2, 10), dims='features')
    it1[Coordinates.Y.value] = xr.DataArray(np.linspace(0, 1, 10), dims='features')

    it2 = IntensityTable.synthetic_intensities(
        codebook,
        num_z=1,
        height=50,
        width=50,
        n_spots=12
    )
    # intensity table 2 has 12 spots, xmin = 1, ymin = 1, xmax = 3, ymax = 3
    it2[Coordinates.X.value] = xr.DataArray(np.linspace(1, 3, 12), dims='features')
    it2[Coordinates.Y.value] = xr.DataArray(np.linspace(1, 3, 12), dims='features')

    IntensityTable.take_max(it1, it2)
