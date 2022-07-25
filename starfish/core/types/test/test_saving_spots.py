import os
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

from starfish.types import Axes, Coordinates, Features
from starfish.core.types import PerImageSliceSpotResults, SpotAttributes, SpotFindingResults
from starfish.core.util.logging import Log

def dummy_spots() -> SpotFindingResults:
    rounds = 4
    channels = 3
    spot_count = 100
    img_dim = {'x': 2048, 'y': 2048, 'z': 29}

    coords = {}
    renameAxes = {
        'x': Coordinates.X.value,
        'y': Coordinates.Y.value,
        'z': Coordinates.Z.value
    }
    for dim in img_dim.keys():
        coords[renameAxes[dim]] = xr.DataArray(np.arange(0, 1, img_dim[dim]))

    log = Log()

    spot_attributes_list = []
    for r in range(rounds):
        for c in range(channels):
            index = {Axes.ROUND: r, Axes.CH: c}
            spots = SpotAttributes(pd.DataFrame(
                np.random.randint(0, 100, size=(spot_count, 4)),
                columns=[Axes.X.value,
                         Axes.Y.value,
                         Axes.ZPLANE.value,
                         Features.SPOT_RADIUS]
            ))
            spot_attributes_list.append(
                (PerImageSliceSpotResults(spots, extras=None), index)
            )

    return SpotFindingResults(
        imagestack_coords=coords,
        log=log,
        spot_attributes_list=spot_attributes_list
    )

def test_saving_spots() -> None:
    data = dummy_spots()

    # test serialization
    tempdir = tempfile.mkdtemp()
    print(tempdir)
    data.save(tempdir + "/")

    # load back into memory
    data2 = SpotFindingResults.load(os.path.join(tempdir, 'SpotFindingResults.json'))

    # ensure all items are equal
    assert data.keys() == data2.keys()
    assert data._log.encode() == data2._log.encode()
    for ax in data.physical_coord_ranges.keys():
        np.testing.assert_equal(data.physical_coord_ranges[ax].to_numpy(),
                                data2.physical_coord_ranges[ax].to_numpy())
    for k in data._results.keys():
        np.testing.assert_array_equal(data._results[k].spot_attrs.data,
                                      data2._results[k].spot_attrs.data)
