import unittest
from collections import OrderedDict

import numpy as np
import xarray as xr

from starfish.imagestack import physical_coordinate_calculator
from starfish.types import Indices, PHYSICAL_COORDINATE_DIMENSION, PhysicalCoordinateTypes


class TestPhysicalCoordinateCalculator(unittest.TestCase):
    """Tests recalculating physical coordinates for an Imagestack with a shape (1, 1, 1, 200, 200)
    and physical coordinates {xmin: 1 , xmax: 2, ymin: 4, ymax: 6, zmin: 1, zmax: 3}"""

    def setUp(self):
        self.coords_array = xr.DataArray(
            np.empty(
                shape=(1, 1, 1, 6),
                dtype=np.float32,
            ),
            dims=(Indices.ROUND.value,
                  Indices.CH.value,
                  Indices.Z.value,
                  PHYSICAL_COORDINATE_DIMENSION),
            coords={
                PHYSICAL_COORDINATE_DIMENSION: [
                    PhysicalCoordinateTypes.X_MIN.value,
                    PhysicalCoordinateTypes.X_MAX.value,
                    PhysicalCoordinateTypes.Y_MIN.value,
                    PhysicalCoordinateTypes.Y_MAX.value,
                    PhysicalCoordinateTypes.Z_MIN.value,
                    PhysicalCoordinateTypes.Z_MAX.value,
                ],
            },
        )

        self.indexers = {Indices.ROUND: slice(None, None),
                         Indices.CH: slice(None, None),
                         Indices.Z: slice(None, None),
                         Indices.Y: slice(None, None),
                         Indices.X: slice(None, None)}

        self.physical_coords = {PhysicalCoordinateTypes.X_MIN: 1,
                                PhysicalCoordinateTypes.X_MAX: 2,
                                PhysicalCoordinateTypes.Y_MIN: 4,
                                PhysicalCoordinateTypes.Y_MAX: 6,
                                PhysicalCoordinateTypes.Z_MIN: 1,
                                PhysicalCoordinateTypes.Z_MAX: 3}

        self.coords_array.loc[0, 0, 0] = \
            np.array([self.physical_coords[PhysicalCoordinateTypes.X_MIN],
                      self.physical_coords[PhysicalCoordinateTypes.X_MAX],
                      self.physical_coords[PhysicalCoordinateTypes.Y_MIN],
                      self.physical_coords[PhysicalCoordinateTypes.Y_MAX],
                      self.physical_coords[PhysicalCoordinateTypes.Z_MIN],
                      self.physical_coords[PhysicalCoordinateTypes.Z_MAX]])

        self.stack_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                        (Indices.Z, 1), (Indices.Y, 200), (Indices.X, 200)])

    def test_calc_new_physical_coords_array(self):
        # Calculate physical pixel sizes for x and y
        physical_pixel_size_x = \
            physical_coordinate_calculator.\
            calculate_physcial_pixel_size(
                coord_max=self.physical_coords[PhysicalCoordinateTypes.X_MAX],
                coord_min=self.physical_coords[PhysicalCoordinateTypes.X_MIN],
                num_pixels=self.stack_shape[Indices.X])

        physical_pixel_size_y = \
            physical_coordinate_calculator.\
            calculate_physcial_pixel_size(
                coord_max=self.physical_coords[PhysicalCoordinateTypes.Y_MAX],
                coord_min=self.physical_coords[PhysicalCoordinateTypes.Y_MIN],
                num_pixels=self.stack_shape[Indices.Y])

        # Index on single value of X, range of y
        self.indexers[Indices.Y], self.indexers[Indices.X] = 100, slice(None, 100)
        new_coords = physical_coordinate_calculator.calc_new_physical_coords_array(
            physical_coordinates=self.coords_array,
            stack_shape=self.stack_shape,
            indexers=self.indexers)

        expected_xmin = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_x,
            index=None,
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.X_MIN])

        expected_xmax = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_x,
            index=101,
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.X_MIN])

        expected_ymin = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_y,
            index=100,
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.Y_MIN])

        expected_ymax = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_y,
            index=101,
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.Y_MIN])
        # z range stays the same
        expected_zmin = self.physical_coords[PhysicalCoordinateTypes.Z_MIN]
        expected_zmax = self.physical_coords[PhysicalCoordinateTypes.Z_MAX]

        assert np.allclose(new_coords.loc[0, 0, 0], np.array([expected_xmin,
                                                              expected_xmax,
                                                              expected_ymin,
                                                              expected_ymax,
                                                              expected_zmin,
                                                              expected_zmax]))

        # Index on last half of Y, first half of X
        self.indexers[Indices.Y], self.indexers[Indices.X] = slice(100, None), slice(None, 100)
        new_coords = physical_coordinate_calculator.calc_new_physical_coords_array(
            physical_coordinates=self.coords_array,
            stack_shape=self.stack_shape,
            indexers=self.indexers)

        expected_xmin = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_x,
            index=None,
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.X_MIN])

        expected_xmax = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_x,
            index=(100 + 1),
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.X_MIN])

        expected_ymin = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_y,
            index=100,
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.Y_MIN])

        expected_ymax = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_y,
            index=None,
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.Y_MIN])

        assert np.allclose(new_coords.loc[0, 0, 0], np.array([expected_xmin,
                                                              expected_xmax,
                                                              expected_ymin,
                                                              expected_ymax,
                                                              expected_zmin,
                                                              expected_zmax]))

        # Negative indexing
        self.indexers[Indices.Y], self.indexers[Indices.X] = slice(100, -10), slice(1, -50)
        new_coords = physical_coordinate_calculator.calc_new_physical_coords_array(
            physical_coordinates=self.coords_array,
            stack_shape=self.stack_shape,
            indexers=self.indexers)

        expected_xmin = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_x,
            index=1,
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.X_MIN])

        expected_xmax = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_x,
            index=(-50 + 1),
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.X_MIN])

        expected_ymin = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_y,
            index=100,
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.Y_MIN])

        expected_ymax = physical_coordinate_calculator.calculate_physical_pixel_value(
            physcial_pixel_size=physical_pixel_size_y,
            index=(-10 + 1),
            start_of_range=self.physical_coords[PhysicalCoordinateTypes.Y_MIN])

        assert np.allclose(new_coords.loc[0, 0, 0], np.array([expected_xmin,
                                                              expected_xmax,
                                                              expected_ymin,
                                                              expected_ymax,
                                                              expected_zmin,
                                                              expected_zmax]))


if __name__ == '__main__':
    unittest.main()
