import pytest
import numpy as np
import pandas as pd
from starfish.types import SpotAttributes, Indices, Features
from starfish import IntensityTable


def spot_attribute_factory(n):
    """construct n synthetic attributes"""
    return SpotAttributes(
        pd.DataFrame(
            data=np.array([[i, i, i, 1] for i in np.arange(n)]),
            columns=[Indices.Z, Indices.Y, Indices.X, Features.SPOT_RADIUS]
        )
    )


def test_intensity_table_can_be_constructed_from_a_numpy_array_and_spot_attributes():
    """Verify that the IntensityTable can be created and that the resulting data is correct."""
    spot_attributes = spot_attribute_factory(3)
    data = np.zeros(30).reshape(3, 5, 2)
    intensities = IntensityTable.from_spot_data(data, spot_attributes)

    assert intensities.shape == data.shape
    assert np.array_equal(intensities.values, data)


def test_from_spot_attributes_must_have_aligned_dimensions_spot_attributes_and_data():
    """
    Number of features must match number of SpotAttributes. Pass two attributes and 3 features and
    verify a ValueError is raised.
    """
    spot_attributes = spot_attribute_factory(2)
    data = np.zeros(30).reshape(3, 5, 2)
    with pytest.raises(ValueError):
        intensities = IntensityTable.from_spot_data(data, spot_attributes)
