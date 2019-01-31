import numpy as np
import xarray as xr

from starfish.imagestack.imagestack import ImageStack
from starfish.imagestack.physical_coordinate_calculator import get_physical_coordinates_of_spot
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
            tile_indices = {
                Axes.ROUND.value: _round,
                Axes.CH.value: ch,
                Axes.ZPLANE.value: pixel_z,
            }
            # Get corresponding physical coordinates
            physical_coords = get_physical_coordinates_of_spot(
                image_stack._coordinates,
                tile_indices,
                pixel_x,
                pixel_y,
                image_stack._tile_shape)
            # Assign to coordinates arrays
            intensity_table[Coordinates.X.value][ind] = physical_coords[0]
            intensity_table[Coordinates.Y.value][ind] = physical_coords[1]
            intensity_table[Coordinates.Z.value][ind] = physical_coords[2]
            break
    return intensity_table
