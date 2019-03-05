import numpy as np
import xarray as xr

from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.types import Axes, Coordinates, Features


def transfer_physical_coords_from_imagestack_to_intensity_table(image_stack: ImageStack,
                                                                intensity_table: IntensityTable
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
    num_features = intensity_table.sizes[Features.AXIS]
    intensity_table[Coordinates.X.value] = xr.DataArray(np.zeros(num_features, np.float32),
                                                        dims='features')
    intensity_table[Coordinates.Y.value] = xr.DataArray(np.zeros(num_features, np.float32),
                                                        dims='features')
    intensity_table[Coordinates.Z.value] = xr.DataArray(np.zeros(num_features, np.float32),
                                                        dims='features')
    for ind, spot in intensity_table.groupby(Features.AXIS):
        # Iterate through r, ch per spot
        for ch, _round in np.ndindex(spot.data.shape):
            # if non zero value set coords
            if spot[ch][_round].data == 0:
                continue
            # get pixel coords of this tile
            pixel_x = spot.coords[Axes.X].data
            pixel_y = spot.coords[Axes.Y].data
            pixel_z = spot.coords[Axes.ZPLANE].data
            # Grab physical coordinates from imagestack.coords
            physical_x = image_stack.xarray[Coordinates.X.value][pixel_x]
            physical_y = image_stack.xarray[Coordinates.Y.value][pixel_y]
            physical_z = image_stack.xarray[Coordinates.Z.value][pixel_z]
            # Assign to coordinates arrays
            intensity_table[Coordinates.X.value][ind] = physical_x
            intensity_table[Coordinates.Y.value][ind] = physical_y
            intensity_table[Coordinates.Z.value][ind] = physical_z
            break
    return intensity_table
