import os
import tempfile

import numpy as np
from starfish import ImageStack, IntensityTable


# fixtures required: Any intensity Table

# can save an intensity table, load it back up, matches original data


def test_intensity_table_serialization():
    data = np.zeros(100).reshape(1, 5, 2, 2, 5)
    image_stack = ImageStack.from_numpy_array(data)
    intensities = IntensityTable.from_image_stack(image_stack)

    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test.nc')

    intensities.save(filename)

    loaded = intensities.load(filename)

    assert intensities.equals(loaded)
