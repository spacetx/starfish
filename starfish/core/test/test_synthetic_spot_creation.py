import numpy as np
import pytest
import xarray as xr

from starfish.core.intensity_table.test.factories import synthetic_intensity_table
from starfish.core.test.factories import synthetic_spot_pass_through_stack, SyntheticData
from starfish.core.types import Axes


def test_synthetic_spot_creation_raises_error_with_coords_too_small():
    num_z = 0
    height = 40
    width = 50
    intensity_table = synthetic_intensity_table()
    with pytest.raises(ValueError):
        SyntheticData.synthetic_spots(intensity_table, num_z, height, width)


def test_synthetic_spot_creation_produces_an_imagestack_with_correct_spot_location():

    codebook, true_intensities, image = synthetic_spot_pass_through_stack()

    g, r, c = np.where(true_intensities.values)

    x = np.empty_like(g)
    y = np.empty_like(g)
    z = np.empty_like(g)
    breaks = np.concatenate([
        np.array([0]),
        np.where(np.diff(g))[0] + 1,
        np.array([g.shape[0]])
    ])
    for i in np.arange(len(breaks) - 1):
        x[breaks[i]: breaks[i + 1]] = true_intensities.coords[Axes.X.value][i]
        y[breaks[i]: breaks[i + 1]] = true_intensities.coords[Axes.Y.value][i]
        z[breaks[i]: breaks[i + 1]] = true_intensities.coords[Axes.ZPLANE.value][i]

    # only 8 values should be set, since there are only 8 locations across the tensor
    assert np.sum(image.xarray != 0) == 8

    intensities = image.xarray.sel(
        x=xr.DataArray(x, dims=['intensity']),
        y=xr.DataArray(y, dims=['intensity']),
        z=xr.DataArray(z, dims=['intensity']),
        r=xr.DataArray(r, dims=['intensity']),
        c=xr.DataArray(c, dims=['intensity']))
    assert np.allclose(
        intensities,
        true_intensities.values[np.where(true_intensities)])
