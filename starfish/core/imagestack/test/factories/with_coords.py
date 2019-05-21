from collections import OrderedDict

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.imagestack.physical_coordinates import _get_physical_coordinates_of_z_plane
from starfish.core.types import Axes, Coordinates, PhysicalCoordinateTypes
from .synthetic_stack import synthetic_stack


def imagestack_with_coords_factory(stack_shape: OrderedDict, coords: OrderedDict) -> ImageStack:
    """
    Create an ImageStack of given shape and assigns the given x,y,z
    min/max physical coordinates to each tile.

    Parameters
    ----------
    stack_shape: OrderedDict
        Dict[Axes, int] defining the size of each dimension for an ImageStack

    coords: OrderedDict
        Dict[PhysicalCoordinateTypes, float] defining the min/max values of physical
        coordinates to assign to the Imagestack
    """

    stack = synthetic_stack(
        num_round=stack_shape[Axes.ROUND],
        num_ch=stack_shape[Axes.CH],
        num_z=stack_shape[Axes.ZPLANE],
        tile_height=stack_shape[Axes.Y],
        tile_width=stack_shape[Axes.X],
    )

    stack.xarray[Coordinates.X.value] = xr.DataArray(
        np.linspace(coords[PhysicalCoordinateTypes.X_MIN], coords[PhysicalCoordinateTypes.X_MAX],
                    stack.xarray.sizes[Axes.X.value]), dims=Axes.X.value)

    stack.xarray[Coordinates.Y.value] = xr.DataArray(
        np.linspace(coords[PhysicalCoordinateTypes.Y_MIN], coords[PhysicalCoordinateTypes.Y_MAX],
                    stack.xarray.sizes[Axes.Y.value]), dims=Axes.Y.value)

    z_coord = _get_physical_coordinates_of_z_plane(
        (coords[PhysicalCoordinateTypes.Z_MIN], coords[PhysicalCoordinateTypes.Z_MAX]))

    stack.xarray[Coordinates.Z.value] = xr.DataArray(np.zeros(
        stack.xarray.sizes[Axes.ZPLANE.value]),
        dims=Axes.ZPLANE.value)

    for z in stack.axis_labels(Axes.ZPLANE):
        stack.xarray[Coordinates.Z.value].loc[z] = z_coord

    return stack
