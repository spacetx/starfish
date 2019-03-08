from copy import deepcopy
from functools import lru_cache

import numpy as np

from starfish.display import (
    _mask_low_intensity_spots,
    _max_intensity_table_maintain_dims,
    _spots_to_markers,
)
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.test import test_utils
from starfish.types import Axes, Features


@lru_cache(maxsize=1)
def testing_data():
    np.random.seed(1)
    codebook = test_utils.codebook_array_factory()
    num_z, height, width = 3, 4, 5
    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=num_z,
        height=height,
        width=width,
        n_spots=4
    )
    return intensities


def test_max_intensity_maintains_each_combination_of_dimensions():
    # Frozen(OrderedDict([('features', 4), ('c', 3), ('r', 2)]))
    intensities = testing_data()
    results = _max_intensity_table_maintain_dims(intensities, {Axes.CH})
    assert results.shape == (4, 1, 2)
    results = _max_intensity_table_maintain_dims(intensities, {Axes.ROUND})
    assert results.shape == (4, 3, 1)
    results = _max_intensity_table_maintain_dims(intensities, {Axes.ROUND, Axes.CH})
    assert results.shape == (4, 1, 1)
    results = _max_intensity_table_maintain_dims(intensities, {Features.AXIS, Axes.CH})
    assert results.shape == (1, 1, 2)
    results = _max_intensity_table_maintain_dims(
        intensities, {Axes.ROUND, Axes.CH, Features.AXIS}
    )
    assert results.shape == (1, 1, 1)


def test_mask_low_intensities_masks_preexisting_nans():
    intensities = testing_data()
    masked = _mask_low_intensity_spots(intensities, intensity_threshold=0)
    # intensity_threshold == 0 should mean no spots are masked
    assert np.sum(masked) == np.product(intensities.shape)

    # with c-order ravel, this nan is in the 5th position (zero-based = 4)
    intensities = deepcopy(intensities)
    intensities[0, 2, 0] = np.nan
    masked = _mask_low_intensity_spots(intensities, intensity_threshold=0)
    assert np.array_equal(np.where(~masked)[0], np.array([4]))


def test_mask_low_intensities():
    intensities = testing_data()
    masked = _mask_low_intensity_spots(intensities, intensity_threshold=0.1)
    expected = np.where(np.ravel(intensities.values >= 0.1))
    passing = np.where(masked)
    assert np.array_equal(passing, expected)


def test_spots_to_markers():
    intensities = testing_data()
    markers, sizes = _spots_to_markers(intensities)
    assert len(markers) == np.product(intensities.shape)

    # check that intensities are being flattened into markers with r rotating fastest, then c
    assert np.all(markers[:2, 2] == [0, 1])  # rounds
    assert np.all(markers[:2, 3] == [0, 0])  # channels
