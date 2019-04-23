import numpy as np
import xarray as xr

from starfish import IntensityTable
from starfish.core.codebook.test.factories import codebook_array_factory, loaded_codebook
from starfish.core.types import Coordinates, Features
from ..overlap import Area


def synthetic_intensity_table() -> IntensityTable:
    return IntensityTable.synthetic_intensities(loaded_codebook(), n_spots=2)


def create_intensity_table_with_coords(area: Area, n_spots: int=10) -> IntensityTable:
    """
    Creates a 50X50 intensity table with physical coordinates within
    the given Area.

    Parameters
    ----------
    area: Area
        The area of physical space the IntensityTable should be defined over
    n_spots:
        Number of spots to add to the IntensityTable
    """
    codebook = codebook_array_factory()
    it = IntensityTable.synthetic_intensities(
        codebook,
        num_z=1,
        height=50,
        width=50,
        n_spots=n_spots
    )
    # intensity table 1 has 10 spots, xmin = 0, ymin = 0, xmax = 2, ymax = 1
    it[Coordinates.X.value] = xr.DataArray(np.linspace(area.min_x, area.max_x, n_spots),
                                           dims=Features.AXIS)
    it[Coordinates.Y.value] = xr.DataArray(np.linspace(area.min_y, area.max_y, n_spots),
                                           dims=Features.AXIS)
    return it
