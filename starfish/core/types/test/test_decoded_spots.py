import os
import tempfile

import pandas as pd
import pytest

from starfish import IntensityTable
from starfish.core.codebook.test.factories import codebook_array_factory
from starfish.core.types import Coordinates, DecodedSpots, Features


def dummy_intensities() -> IntensityTable:

    codebook = codebook_array_factory()
    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=10,
        height=10,
        width=10,
        n_spots=5,
    )

    intensities[Coordinates.Z.value] = (Features.AXIS, [0, 1, 0, 1, 0])
    intensities[Coordinates.Y.value] = (Features.AXIS, [10, 30, 50, 40, 20])
    intensities[Coordinates.X.value] = (Features.AXIS, [50.2, 30.2, 60.2, 40.2, 70.2])

    # remove target from dummy to test error messages
    del intensities[Features.TARGET]

    return intensities


def test_decoded_spots() -> None:
    data = dummy_intensities()

    with pytest.raises(RuntimeError):
        data.to_decoded_spots()

    # mock decoder run by adding target list
    data[Features.TARGET] = (Features.AXIS, list('abcde'))

    ds = data.to_decoded_spots()

    assert ds.data.shape[0] == 5

    # test serialization
    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'spots.csv')
    ds.save_csv(filename)

    # load back into memory
    ds2 = DecodedSpots.load_csv(filename)
    pd.testing.assert_frame_equal(ds.data, ds2.data)
