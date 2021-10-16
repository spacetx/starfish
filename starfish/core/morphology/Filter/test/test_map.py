import numpy as np

from starfish.core.morphology.binary_mask.test.factories import binary_mask_collection_2d
from starfish.core.types import Axes, FunctionSource
from ..map import Map


def test_apply():
    input_mask_collection = binary_mask_collection_2d()
    filt = Map("morphology.binary_dilation", module=FunctionSource.skimage)
    output_mask_collection = filt.run(input_mask_collection)

    assert input_mask_collection._pixel_ticks == output_mask_collection._pixel_ticks
    assert input_mask_collection._physical_ticks == output_mask_collection._physical_ticks
    assert input_mask_collection._log == output_mask_collection._log
    assert len(input_mask_collection) == len(output_mask_collection)

    region_0, region_1 = output_mask_collection.masks()

    assert region_0.name == '0'
    assert region_1.name == '1'

    temp = np.ones((2, 6), dtype=bool)
    assert np.array_equal(region_0, temp)
    temp = np.ones((3, 4), dtype=bool)
    temp[0, 0] = 0
    assert np.array_equal(region_1, temp)

    assert np.array_equal(region_0[Axes.Y.value], [0, 1])
    assert np.array_equal(region_0[Axes.X.value], [0, 1, 2, 3, 4, 5])

    assert np.array_equal(region_1[Axes.Y.value], [2, 3, 4])
    assert np.array_equal(region_1[Axes.X.value], [2, 3, 4, 5])
