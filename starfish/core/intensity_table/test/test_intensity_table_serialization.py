"""
Tests for IntensityTable.save and IntensityTable.load methods
"""

import os
import tempfile

import numpy as np

from starfish import ImageStack
from ..intensity_table import IntensityTable


def test_intensity_table_serialization():
    """
    Test that an IntensityTable can be saved to disk, and that when it is reloaded, the data is
    unchanged
    """

    # create an IntensityTable
    data = np.zeros(100, dtype=np.float32).reshape(1, 5, 2, 2, 5)
    image_stack = ImageStack.from_numpy(data)
    intensities = IntensityTable.from_image_stack(image_stack)

    # dump it to disk
    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test.nc')
    intensities.to_netcdf(filename)

    # verify the data has not changed
    loaded = intensities.open_netcdf(filename)
    assert intensities.equals(loaded)
