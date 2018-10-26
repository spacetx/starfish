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
            dims=('r', 'c', 'z', 'physical_coordinate'),
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

        self.indexers = {'r': slice(None, None),
                         'c': slice(None, None),
                         'z': slice(None, None),
                         'y': slice(None, None),
                         'x': slice(None, None)}

        self.coords_array.loc[0, 0, 0] = np.array([1, 2, 4, 6, 1, 3])

        self.stack_shape = OrderedDict([(Indices.ROUND, 1), (Indices.CH, 1),
                                        (Indices.Z, 1), (Indices.Y, 200), (Indices.X, 200)])

    def test_calc_new_physical_coords_array(self):

        self.indexers['y'], self.indexers['x'] = 100, slice(None, 100)
        new_coords = physical_coordinate_calculator.calc_new_physical_coords_array(
            self.coords_array, self.stack_shape, self.indexers)
        assert np.allclose(new_coords.loc[0, 0, 0], np.array([1, 1.505, 5, 5.01, 1, 3]))

        # xmin = 1, xmax = 1.505 (+ .005 size of one x physical pixel)
        # ymin = 4, ymax=5.01 (+ .01 size of one y physical pixel)
        self.indexers['y'], self.indexers['x'] = slice(None, 100), slice(None, 100)
        new_coords = physical_coordinate_calculator.calc_new_physical_coords_array(
            self.coords_array, self.stack_shape, self.indexers)
        assert np.allclose(new_coords.loc[0, 0, 0], np.array([1, 1.505, 4, 5.01, 1, 3]))

        self.indexers['y'], self.indexers['x'] = 100, 150
        new_coords = physical_coordinate_calculator.calc_new_physical_coords_array(
            self.coords_array, self.stack_shape, self.indexers)
        assert np.allclose(new_coords.loc[0, 0, 0], np.array([1.75, 1.755, 5, 5.01, 1, 3]))


if __name__ == '__main__':
    unittest.main()
