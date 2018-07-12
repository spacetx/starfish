import numpy as np
import pandas as pd

from starfish.constants import Indices
from starfish.intensity_table import IntensityTable
# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import (
    loaded_codebook, simple_codebook_json, simple_codebook_array)


def test_empty_intensity_table():
    x = [1, 2]
    y = [2, 3]
    z = [1, 1]
    r = [1, 1]
    spot_attributes = pd.MultiIndex.from_arrays([x, y, z, r], names=('x', 'y', 'z', 'r'))
    empty = IntensityTable.empty_intensity_table(spot_attributes, 2, 2)
    assert empty.shape == (2, 2, 2)
    assert np.sum(empty.values) == 0


def test_synthetic_intensities_generates_correct_number_of_features(loaded_codebook):
    n_spots = 2
    intensities = IntensityTable.synthetic_intensities(loaded_codebook, n_spots=n_spots)
    assert isinstance(intensities, IntensityTable)

    # shape should have n_spots and channels and hybridization rounds equal to the codebook's shape
    assert intensities.shape == (n_spots, *loaded_codebook.shape[1:])


def test_synthetic_intensities_have_correct_number_of_on_features(loaded_codebook):
    n_spots = 2
    intensities = IntensityTable.synthetic_intensities(loaded_codebook, n_spots=n_spots)
    on_features = np.sum(intensities.values != 0)
    # this asserts that the number of features "on" in intensities should be equal to the
    # number of "on" features in the codewords, times the total number of spots in intensities.
    assert on_features == loaded_codebook.sum((Indices.CH, Indices.HYB)).values.mean() * n_spots
