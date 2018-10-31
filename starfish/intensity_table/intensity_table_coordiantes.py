import numpy as np
import xarray as xr

from starfish.imagestack import physical_coordinate_calculator
from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.types import Coordinates, Features, Indices


def transfer_physical_coords_from_imagestack_to_intensity_table(image_stack: ImageStack,
                                                                intensity_table: IntensityTable
                                                                ) -> IntensityTable:
    """Transfers physical coordinates from an imagestacks coordianates xarray to an intensity table
        1.) Creates three new coords on the intensity table (xc, yc, zc)
        2.) For every spot:
                Get pixel x,y values
                Calculate the physical x,y values
                Assign those values to the coords arrays for this spot
    """""
    # Add three new coords to xarray (xc, yc, zc)
    intensity_table[Coordinates.X.value] = \
        xr.DataArray.astype(intensity_table.features * 0, np.float32)
    intensity_table[Coordinates.Y.value] = \
        xr.DataArray.astype(intensity_table.features * 0, np.float32)
    intensity_table[Coordinates.Z.value] = \
        xr.DataArray.astype(intensity_table.features * 0, np.float32)
    # Iterate through spots
    for ind, spot in intensity_table.groupby(Features.AXIS):
        # Iterate through r, ch per spot
        for ch, round in np.ndindex(spot.data.shape):
            # if non zero value set coords
            if spot[ch][round].data > 0:
                # get pixel coords of this tile
                pixel_x = spot.coords[Indices.X].data
                pixel_y = spot.coords[Indices.Y].data
                pixel_z = spot.coords[Indices.Z].data
                tile_indices = {
                    Indices.ROUND.value: round,
                    Indices.CH.value: ch,
                    Indices.Z.value: pixel_z,
                }
                # Get cooresponding physical coords
                physical_coords = physical_coordinate_calculator.\
                    get_physcial_coordinates_of_spot(image_stack._coordinates,
                                                     tile_indices,
                                                     pixel_x,
                                                     pixel_y,
                                                     image_stack._tile_shape)
                # Assign to coords array
                intensity_table[Coordinates.X.value][ind] = physical_coords[0]
                intensity_table[Coordinates.Y.value][ind] = physical_coords[1]
                intensity_table[Coordinates.Z.value][ind] = physical_coords[2]
                break
    return intensity_table
