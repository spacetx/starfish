import random
from collections import OrderedDict
from typing import Union

import numpy as np

from starfish import IntensityTable
from starfish.intensity_table import intensity_table_coordinates
from starfish.test import factories
from starfish.types import Axes, Coordinates, Features, PhysicalCoordinateTypes

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

    stack = factories.imagestack_with_coords_factory(stack_shape, physical_coords)
    codebook = factories.codebook_array_factory()

    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=stack_shape[Axes.ZPLANE],
        height=stack_shape[Axes.Y],
        width=stack_shape[Axes.X],
        n_spots=NUMBER_SPOTS
    )

    intensities = intensity_table_coordinates.\
        transfer_physical_coords_from_imagestack_to_intensity_table(stack, intensities)

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
        physical_x = stack.xarray[Coordinates.X.value][pixel_x]
        assert np.isclose(spot[Coordinates.X.value], physical_x)

    for spot in yc.features:
        pixel_y = spot[Axes.Y.value].data
        physical_y = stack.xarray[Coordinates.Y.value][pixel_y]
        assert np.isclose(spot[Coordinates.Y.value], physical_y)

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

    stack = factories.imagestack_with_coords_factory(stack_shape, physical_coords)
    codebook = factories.codebook_array_factory()

    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=stack_shape[Axes.ZPLANE],
        height=stack_shape[Axes.Y],
        width=stack_shape[Axes.X],
        n_spots=NUMBER_SPOTS
    )

    intensities = intensity_table_coordinates. \
        transfer_physical_coords_from_imagestack_to_intensity_table(stack, intensities)

    # Check that error is thrown before target assignment
    try:
        intensities.to_expression_matrix()
    except KeyError as e:
        # Assert value error is thrown with right message
        assert e.args[0] == "IntensityTable must have 'cell_id' assignments for each cell before " \
                            "this function can be called. See starfish.TargetAssignment.Label."

    # mock out come cell_ids
    cell_ids = random.sample(range(1, 20), NUMBER_SPOTS)
    intensities[Features.CELL_ID] = (Features.AXIS, cell_ids)

    expression_matrix = intensities.to_expression_matrix()
    # Assert that coords were transferred
    xc = expression_matrix.coords[Coordinates.X]
    yc = expression_matrix.coords[Coordinates.Y]
    zc = expression_matrix.coords[Coordinates.Z]
    assert xc.size == len(set(cell_ids))
    assert yc.size == len(set(cell_ids))
    assert zc.size == len(set(cell_ids))
