"""
Tests for IntensityTable.empty_intensity_table method
"""

import numpy as np
import pandas as pd
import pytest

from starfish.core.types import Axes, Features, SpotAttributes
from ..intensity_table import IntensityTable


def test_intensity_table_can_be_created_from_spot_attributes():
    """
    This test creates an IntensityTable from spot attributes, and verifies that the size matches
    what was requested and that the values are all zero.
    """

    # input has two spots
    spot_attributes = SpotAttributes(
        pd.DataFrame(
            data=np.array(
                [[1, 1, 1, 1],
                 [2, 2, 2, 1]]
            ),
            columns=[Axes.ZPLANE, Axes.Y, Axes.X, Features.SPOT_RADIUS]
        )
    )

    intensities = IntensityTable.zeros(
        spot_attributes,
        round_labels=np.arange(3),
        ch_labels=np.arange(1),
    )

    assert intensities.sizes[Axes.CH] == 1
    assert intensities.sizes[Axes.ROUND] == 3
    assert intensities.sizes[Features.AXIS] == 2
    assert np.all(intensities.values == 0)


def test_from_spot_attributes_throws_type_error_when_passed_a_dataframe():
    """SpotAttributes should be passed instead."""
    # input has two spots
    not_spot_attributes = pd.DataFrame(
        data=np.array(
            [[1, 1, 1, 1],
             [2, 2, 2, 1]]
        ),
        columns=[Axes.ZPLANE, Axes.Y, Axes.X, Features.SPOT_RADIUS]
    )

    with pytest.raises(TypeError):
        IntensityTable.zeros(not_spot_attributes, n_ch=1, n_round=3)
