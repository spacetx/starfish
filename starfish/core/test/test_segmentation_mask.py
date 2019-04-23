import os

import numpy as np
import pytest
import xarray as xr

from starfish.core.segmentation_mask import (
    SegmentationMaskCollection,
    validate_segmentation_mask,
)
from starfish.core.types import Axes, Coordinates


def test_validate_segmentation_mask():
    good = xr.DataArray([[True, False, False],
                         [False, True, True]],
                        dims=('y', 'x'),
                        coords=dict(x=[0, 1, 2],
                                    y=[0, 1],
                                    xc=('x', [0.5, 1.5, 2.5]),
                                    yc=('y', [0.5, 1.5])))
    validate_segmentation_mask(good)

    good = xr.DataArray([[[True], [False], [False]],
                         [[False], [True], [True]]],
                        dims=('z', 'y', 'x'),
                        coords=dict(z=[0, 1],
                                    y=[1, 2, 3],
                                    x=[42],
                                    zc=('z', [0.5, 1.5]),
                                    yc=('y', [1.5, 2.5, 3.5]),
                                    xc=('x', [42.5])))
    validate_segmentation_mask(good)

    bad = xr.DataArray([[1, 2, 3],
                        [4, 5, 6]],
                       dims=('y', 'x'),
                       coords=dict(x=[0, 1, 2],
                                   y=[0, 1],
                                   xc=('x', [0.5, 1.5, 2.5]),
                                   yc=('y', [0.5, 1.5])))
    with pytest.raises(TypeError):
        validate_segmentation_mask(bad)

    bad = xr.DataArray([True],
                       dims=('x'),
                       coords=dict(x=[0],
                                   xc=('x', [0.5])))
    with pytest.raises(TypeError):
        validate_segmentation_mask(bad)

    bad = xr.DataArray([[True]],
                       dims=('z', 'y'),
                       coords=dict(z=[0],
                                   y=[0],
                                   zc=('z', [0.5]),
                                   yc=('y', [0.5])))
    with pytest.raises(TypeError):
        validate_segmentation_mask(bad)

    bad = xr.DataArray([[True]],
                       dims=('x', 'y'))
    with pytest.raises(TypeError):
        validate_segmentation_mask(bad)


def test_from_label_image():
    label_image = np.zeros((5, 5), dtype=np.int32)
    label_image[0] = 1
    label_image[3:5, 3:5] = 2
    label_image[-1, -1] = 0

    physical_ticks = {Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
                      Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12]}

    masks = SegmentationMaskCollection.from_label_image(label_image,
                                                        physical_ticks)

    assert len(masks) == 2

    region_1, region_2 = masks

    assert region_1.name == '1'
    assert region_2.name == '2'

    assert np.array_equal(region_1, np.ones((1, 5), dtype=np.bool))
    temp = np.ones((2, 2), dtype=np.bool)
    temp[-1, -1] = False
    assert np.array_equal(region_2, temp)

    assert np.array_equal(region_1[Axes.Y.value], [0])
    assert np.array_equal(region_1[Axes.X.value], [0, 1, 2, 3, 4])

    assert np.array_equal(region_2[Axes.Y.value], [3, 4])
    assert np.array_equal(region_2[Axes.X.value], [3, 4])

    assert np.array_equal(region_1[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][0:1])
    assert np.array_equal(region_1[Coordinates.X.value],
                          physical_ticks[Coordinates.X][0:5])

    assert np.array_equal(region_2[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][3:5])
    assert np.array_equal(region_2[Coordinates.X.value],
                          physical_ticks[Coordinates.X][3:5])


def test_save_load():
    label_image = np.zeros((5, 5), dtype=np.int32)
    label_image[0] = 1
    label_image[3:5, 3:5] = 2
    label_image[-1, -1] = 0

    physical_ticks = {Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
                      Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12]}

    masks = SegmentationMaskCollection.from_label_image(label_image,
                                                        physical_ticks)

    path = 'data.tgz'
    try:
        masks.save(path)
        masks2 = SegmentationMaskCollection.from_disk(path)
        for m, m2 in zip(masks, masks2):
            assert np.array_equal(m, m2)
    finally:
        os.remove(path)
