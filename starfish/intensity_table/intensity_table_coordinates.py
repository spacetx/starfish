import xarray as xr

from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.types import Axes, Coordinates, Features


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
    for px, coord in pairs:
        pixels = image_stack.xarray[px].values
        feature_inds = intensity_table[px].values.astype(int)
        pixel_inds = pixels[feature_inds]
        coordinates = image_stack.xarray[coord][pixel_inds]
        intensity_table[coord] = xr.DataArray(coordinates.values, dims='features')

    return intensity_table
