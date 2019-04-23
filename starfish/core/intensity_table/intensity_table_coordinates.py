import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Coordinates, Features


def transfer_physical_coords_from_imagestack_to_intensity_table(
        image_stack: ImageStack, intensity_table: IntensityTable
) -> IntensityTable:
    """
    Transfers physical coordinates from an Imagestack's coordinates xarray to an intensity table

    1. Creates three new coords on the intensity table (xc, yc, zc)
    2. For every spot:
        - Get pixel x,y values
        - Calculate the physical x,y values
        - Assign those values to the coords arrays for this spot
    """
    # TODO shanaxel42 consider refactoring pixel case where were can just reshape from Imagestack
    # Add three new coords to xarray (xc, yc, zc)
    if intensity_table.sizes[Features.AXIS] == 0:
        return intensity_table

    pairs = (
        (Axes.X.value, Coordinates.X.value),
        (Axes.Y.value, Coordinates.Y.value),
        (Axes.ZPLANE.value, Coordinates.Z.value)
    )
    for axis, coord in pairs:

        imagestack_pixels: np.ndarray = image_stack.xarray[axis].values
        intensity_table_pixels: np.ndarray = intensity_table[axis].values.astype(int)
        pixel_inds: np.ndarray = imagestack_pixels[intensity_table_pixels]
        coordinates: xr.DataArray = image_stack.xarray[coord][pixel_inds]

        intensity_table[coord] = xr.DataArray(coordinates.values, dims='features')

    return intensity_table
