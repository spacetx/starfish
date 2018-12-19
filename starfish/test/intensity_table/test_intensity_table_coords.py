from collections import OrderedDict
from typing import Union

import numpy as np

from starfish import IntensityTable
from starfish.imagestack import physical_coordinate_calculator
from starfish.intensity_table import intensity_table_coordinates
from starfish.test import test_utils
from starfish.types import Coordinates, Indices, PhysicalCoordinateTypes


NUMBER_SPOTS = 10


def physical_cord_to_pixel_value(physical_coord: float,
                                 physical_pixel_size: Union[float, int],
                                 coordinates_at_pixel_offset_0: int):

    return (physical_coord - coordinates_at_pixel_offset_0) / physical_pixel_size


def test_tranfering_physical_coords_to_intensity_table():
    stack_shape = OrderedDict([(Indices.ROUND, 3), (Indices.CH, 2),
                               (Indices.Z, 1), (Indices.Y, 50), (Indices.X, 40)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, 1),
                                   (PhysicalCoordinateTypes.X_MAX, 2),
                                   (PhysicalCoordinateTypes.Y_MIN, 4),
                                   (PhysicalCoordinateTypes.Y_MAX, 6),
                                   (PhysicalCoordinateTypes.Z_MIN, 1),
                                   (PhysicalCoordinateTypes.Z_MAX, 3)])

    stack = test_utils.imagestack_with_coords_factory(stack_shape, physical_coords)
    codebook = test_utils.codebook_array_factory()

    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=stack_shape[Indices.Z],
        height=stack_shape[Indices.Y],
        width=stack_shape[Indices.X],
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

    physical_pixel_size_x = physical_coordinate_calculator._calculate_physical_pixel_size(
        coord_min=physical_coords[PhysicalCoordinateTypes.X_MIN],
        coord_max=physical_coords[PhysicalCoordinateTypes.X_MAX],
        num_pixels=stack_shape[Indices.X])

    physical_pixel_size_y = physical_coordinate_calculator._calculate_physical_pixel_size(
        coord_min=physical_coords[PhysicalCoordinateTypes.Y_MIN],
        coord_max=physical_coords[PhysicalCoordinateTypes.Y_MAX],
        num_pixels=stack_shape[Indices.Y])

    # Assert that the physical coords align with their corresponding pixel coords
    for spot in xc.features:
        pixel_x = spot[Indices.X.value].data
        physical_x = spot[Coordinates.X.value].data
        calculated_pixel = physical_cord_to_pixel_value(physical_x,
                                                        physical_pixel_size_x,
                                                        physical_coords[
                                                            PhysicalCoordinateTypes.X_MIN
                                                        ])
        assert np.isclose(pixel_x, calculated_pixel)

    for spot in yc.features:
        pixel_y = spot[Indices.Y.value].data
        physical_y = spot[Coordinates.Y.value].data
        calculated_pixel = physical_cord_to_pixel_value(physical_y,
                                                        physical_pixel_size_y,
                                                        physical_coords[
                                                            PhysicalCoordinateTypes.Y_MIN
                                                        ])
        assert np.isclose(pixel_y, calculated_pixel)

    # Assert that zc value is middle of z range
    for spot in zc.features:
        physical_z = spot[Coordinates.Z.value].data
        assert np.isclose(physical_coords[PhysicalCoordinateTypes.Z_MAX],
                          (physical_z * 2) - physical_coords[PhysicalCoordinateTypes.Z_MIN])
