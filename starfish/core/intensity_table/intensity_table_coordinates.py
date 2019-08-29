from typing import Optional

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Coordinates, Features, SpotFindingResults


def transfer_physical_coords_to_intensity_table(intensity_table: IntensityTable,
                                                image_stack: Optional[ImageStack] = None,
                                                spots: Optional[SpotFindingResults] = None,
                                                ) -> IntensityTable:
    """
    Transfers physical coordinates from either an Imagestack or SpotFindingResults to an intensity table

    1. Creates three new coords on the intensity table (xc, yc, zc)
    2. For every spot:
        - Get pixel x,y values
        - Calculate the physical x,y values
        - Assign those values to the coords arrays for this spot
    """
    # TODO shanaxel42 consider refactoring pixel case where were can just reshape from Imagestack

    pairs = (
        (Axes.X.value, Coordinates.X.value),
        (Axes.Y.value, Coordinates.Y.value),
        (Axes.ZPLANE.value, Coordinates.Z.value)
    )

    if spots:
        coord_ranges = spots.physical_coord_ranges
    else:
        coord_ranges = {
            Axes.X: image_stack.xarray[Coordinates.X.value],
            Axes.Y: image_stack.xarray[Coordinates.Y.value],
            Axes.ZPLANE: image_stack.xarray[Coordinates.Z.value]
        }

    # make sure the intensity table gets empty metadata if there are no intensities
    if intensity_table.sizes[Features.AXIS] == 0:
        for axis, coord in pairs:
            intensity_table[coord] = xr.DataArray(np.zeros((0)), dims=Features.AXIS)
        return intensity_table

    for axis, coord in pairs:
        imagestack_pixels: xr.DataArray = coord_ranges[axis]
        intensity_table_pixel_offsets: np.ndarray = intensity_table[axis].values

        # can't interpolate if the axis size == 1, so just select in that case.
        if len(imagestack_pixels) == 1:
            coordinate_fetcher = imagestack_pixels.sel
        else:
            coordinate_fetcher = imagestack_pixels.interp
        coordinates = coordinate_fetcher({axis: intensity_table_pixel_offsets})[coord]

        intensity_table[coord] = xr.DataArray(coordinates.values, dims='features')

    return intensity_table


