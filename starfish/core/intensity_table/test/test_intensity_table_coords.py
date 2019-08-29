import math
import random
from collections import OrderedDict
from typing import Union

import numpy as np

from starfish.core.codebook.test.factories import codebook_array_factory
from starfish.core.imagestack.test.factories import imagestack_with_coords_factory
from starfish.core.types import Axes, Coordinates, Features, PhysicalCoordinateTypes
from .factories import synthetic_decoded_intensity_table
from ..intensity_table import IntensityTable
from ..intensity_table_coordinates import (
    transfer_physical_coords_to_intensity_table,
)

NUMBER_SPOTS = 10


def physical_cord_to_pixel_value(physical_coord: float,
                                 physical_pixel_size: Union[float, int],
                                 coordinates_at_pixel_offset_0: int):

    return (physical_coord - coordinates_at_pixel_offset_0) / physical_pixel_size


def test_tranfering_physical_coords_to_intensity_table():
    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 1), (Axes.Y, 50), (Axes.X, 40)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, 1),
                                   (PhysicalCoordinateTypes.X_MAX, 2),
                                   (PhysicalCoordinateTypes.Y_MIN, 4),
                                   (PhysicalCoordinateTypes.Y_MAX, 6),
                                   (PhysicalCoordinateTypes.Z_MIN, 1),
                                   (PhysicalCoordinateTypes.Z_MAX, 3)])

    stack = imagestack_with_coords_factory(stack_shape, physical_coords)
    codebook = codebook_array_factory()

    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=stack_shape[Axes.ZPLANE],
        height=stack_shape[Axes.Y],
        width=stack_shape[Axes.X],
        n_spots=NUMBER_SPOTS
    )

    intensities = transfer_physical_coords_to_intensity_table(intensity_table=intensities,
                                                              image_stack=stack)

    # Assert that new cords were added
    xc = intensities.coords[Coordinates.X]
    yc = intensities.coords[Coordinates.Y]
    zc = intensities.coords[Coordinates.Z]
    assert xc.size == NUMBER_SPOTS
    assert yc.size == NUMBER_SPOTS
    assert zc.size == NUMBER_SPOTS

    # Assert that the physical coords align with their corresponding pixel coords
    for spot in xc.features:
        pixel_x = spot[Axes.X.value].data
        pixel_x_floor, pixel_x_ceiling = math.floor(pixel_x), math.ceil(pixel_x)
        physical_x_floor = stack.xarray[Coordinates.X.value][pixel_x_floor]
        physical_x_ceiling = stack.xarray[Coordinates.X.value][pixel_x_ceiling]
        assert physical_x_floor <= spot[Coordinates.X.value] <= physical_x_ceiling

    for spot in yc.features:
        pixel_y = spot[Axes.Y.value].data
        pixel_y_floor, pixel_y_ceiling = math.floor(pixel_y), math.ceil(pixel_y)
        physical_y_floor = stack.xarray[Coordinates.Y.value][pixel_y_floor]
        physical_y_ceiling = stack.xarray[Coordinates.Y.value][pixel_y_ceiling]
        assert physical_y_floor <= spot[Coordinates.Y.value] <= physical_y_ceiling

    # Assert that zc value is middle of z range
    for spot in zc.features:
        z_plane = spot[Axes.ZPLANE.value].data
        physical_z = stack.xarray[Coordinates.Z.value][z_plane]
        assert np.isclose(spot[Coordinates.Z.value], physical_z)


def test_tranfering_physical_coords_to_expression_matrix():
    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 1), (Axes.Y, 50), (Axes.X, 40)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, 1),
                                   (PhysicalCoordinateTypes.X_MAX, 2),
                                   (PhysicalCoordinateTypes.Y_MIN, 4),
                                   (PhysicalCoordinateTypes.Y_MAX, 6),
                                   (PhysicalCoordinateTypes.Z_MIN, 1),
                                   (PhysicalCoordinateTypes.Z_MAX, 3)])

    stack = imagestack_with_coords_factory(stack_shape, physical_coords)
    codebook = codebook_array_factory()

    decoded_intensities = synthetic_decoded_intensity_table(
        codebook,
        num_z=stack_shape[Axes.ZPLANE],
        height=stack_shape[Axes.Y],
        width=stack_shape[Axes.X],
        n_spots=NUMBER_SPOTS
    )

    intensities = transfer_physical_coords_to_intensity_table(
        image_stack=stack, intensity_table=decoded_intensities)

    # Check that error is thrown before target assignment
    try:
        decoded_intensities.to_expression_matrix()
    except KeyError as e:
        # Assert value error is thrown with right message
        assert e.args[0] == "IntensityTable must have 'cell_id' assignments for each cell before " \
                            "this function can be called. See starfish.spots.AssignTargets.Label."

    # mock out come cell_ids
    cell_ids = random.sample(range(1, 20), NUMBER_SPOTS)
    decoded_intensities[Features.CELL_ID] = (Features.AXIS, cell_ids)

    expression_matrix = intensities.to_expression_matrix()
    # Assert that coords were transferred
    xc = expression_matrix.coords[Coordinates.X]
    yc = expression_matrix.coords[Coordinates.Y]
    zc = expression_matrix.coords[Coordinates.Z]
    assert xc.size == len(set(cell_ids))
    assert yc.size == len(set(cell_ids))
    assert zc.size == len(set(cell_ids))
