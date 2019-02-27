from collections import OrderedDict

import numpy as np
import pytest
import xarray as xr
from slicedimage import ImageFormat

from starfish.imagestack.imagestack import ImageStack
from starfish.imagestack.physical_coordinate_calculator import get_physical_coordinates_of_z_plane
from starfish.intensity_table.intensity_table import IntensityTable
# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test import test_utils
from starfish.test.dataset_fixtures import (  # noqa: F401
    codebook_intensities_image_for_single_synthetic_spot,
    loaded_codebook,
    simple_codebook_array,
    simple_codebook_json,
    synthetic_dataset_with_truth_values,
    synthetic_intensity_table,
    synthetic_spot_pass_through_stack,
)
from starfish.types import Axes, PhysicalCoordinateTypes
from .imagestack_test_utils import verify_physical_coordinates

X_COORDS = 1, 2
Y_COORDS = 4, 6
Z_COORDS = 1, 3


def test_get_slice_simple_index():
    """
    Retrieve a slice across one of the axes at the end.  For instance, if the axes are
    (P, Q0,..., Qn-1, R), slice across either P or R.  This test has expectations regarding the
    ordering of the axes in the ImageStack.
    """
    stack = ImageStack.synthetic_stack()
    round_ = 1
    imageslice, axes = stack.get_slice(
        {Axes.ROUND: round_}
    )
    assert axes == [Axes.CH, Axes.ZPLANE]

    y, x = stack.tile_shape

    for ch in range(stack.shape[Axes.CH]):
        for z in range(stack.shape[Axes.ZPLANE]):
            data = np.empty((y, x))
            data.fill((round_ * stack.shape[Axes.CH] + ch) * stack.shape[Axes.ZPLANE] + z)

            assert data.all() == imageslice[ch, z].all()


def test_get_slice_middle_index():
    """
    Retrieve a slice across one of the axes in the middle.  For instance, if the axes are
    (P, Q0,..., Qn-1, R), slice across one of the Q axes.  This test has expectations regarding the
    ordering of the axes in the ImageStack.
    """
    stack = ImageStack.synthetic_stack()
    ch = 1
    imageslice, axes = stack.get_slice(
        {Axes.CH: ch}
    )
    assert axes == [Axes.ROUND, Axes.ZPLANE]

    y, x = stack.tile_shape

    for round_ in range(stack.shape[Axes.ROUND]):
        for z in range(stack.shape[Axes.ZPLANE]):
            data = np.empty((y, x))
            data.fill((round_ * stack.shape[Axes.CH] + ch) * stack.shape[Axes.ZPLANE] + z)

            assert data.all() == imageslice[round_, z].all()


def test_get_slice_range():
    """
    Retrieve a slice across a range of one of the dimensions.
    """
    stack = ImageStack.synthetic_stack()
    zrange = slice(1, 3)
    imageslice, axes = stack.get_slice(
        {Axes.ZPLANE: zrange}
    )
    y, x = stack.tile_shape
    assert axes == [Axes.ROUND, Axes.CH, Axes.ZPLANE]

    for round_ in range(stack.shape[Axes.ROUND]):
        for ch in range(stack.shape[Axes.CH]):
            for z in range(zrange.stop - zrange.start):
                data = np.empty((y, x))
                data.fill((round_ * stack.shape[Axes.CH] + ch) * stack.shape[Axes.ZPLANE]
                          + (z + zrange.start))

                assert data.all() == imageslice[round_, ch, z].all()


def test_set_slice_simple_index():
    """
    Sets a slice across one of the axes at the end.  For instance, if the axes are
    (P, Q0,..., Qn-1, R), sets a slice across either P or R.  This test has expectations regarding
    the ordering of the axes in the ImageStack.
    """
    stack = ImageStack.synthetic_stack()
    round_ = 1
    y, x = stack.tile_shape

    expected = np.full(
        (stack.shape[Axes.CH], stack.shape[Axes.ZPLANE], y, x),
        fill_value=0.5,
        dtype=np.float32
    )
    index = {Axes.ROUND: round_}

    stack.set_slice(index, expected, [Axes.CH, Axes.ZPLANE])

    assert np.array_equal(stack.get_slice(index)[0], expected)


def test_set_slice_middle_index():
    """
    Sets a slice across one of the axes in the middle.  For instance, if the axes are
    (P, Q0,..., Qn-1, R), slice across one of the Q axes.  This test has expectations regarding the
    ordering of the axes in the ImageStack.
    """
    stack = ImageStack.synthetic_stack()
    ch = 1
    y, x = stack.tile_shape

    expected = np.full(
        (stack.shape[Axes.ROUND], stack.shape[Axes.ZPLANE], y, x),
        fill_value=0.5,
        dtype=np.float32
    )
    index = {Axes.CH: ch}

    stack.set_slice(index, expected, [Axes.ROUND, Axes.ZPLANE])

    assert np.array_equal(stack.get_slice(index)[0], expected)


def test_set_slice_reorder():
    """
    Sets a slice across one of the axes.  The source data is not in the same order as the axes in
    ImageStack, but set_slice should reorder the axes and write it correctly.
    """
    stack = ImageStack.synthetic_stack()
    round_ = 1
    y, x = stack.tile_shape
    index = {Axes.ROUND: round_}

    written = np.full(
        (stack.shape[Axes.ZPLANE], stack.shape[Axes.CH], y, x),
        fill_value=0.5,
        dtype=np.float32
    )
    stack.set_slice(index, written, [Axes.ZPLANE, Axes.CH])

    expected = np.full(
        (stack.shape[Axes.CH], stack.shape[Axes.ZPLANE], y, x),
        fill_value=0.5,
        dtype=np.float32
    )
    assert np.array_equal(stack.get_slice(index)[0], expected)


def test_set_slice_range():
    """
    Sets a slice across a range of one of the axes.
    """
    stack = ImageStack.synthetic_stack()
    zrange = slice(1, 3)
    y, x = stack.tile_shape

    expected = np.full(
        (stack.shape[Axes.ROUND], stack.shape[Axes.CH], zrange.stop - zrange.start + 1, y, x),
        fill_value=0.5,
        dtype=np.float32
    )
    index = {Axes.ZPLANE: zrange}

    stack.set_slice(index, expected, [Axes.ROUND, Axes.CH, Axes.ZPLANE])

    assert np.array_equal(stack.get_slice(index)[0], expected)


def test_from_numpy_array_raises_error_when_incorrect_dims_passed():
    array = np.ones((2, 2), dtype=np.float32)
    # verify this method works with the correct shape
    image = ImageStack.from_numpy_array(array.reshape((1, 1, 1, 2, 2)))
    assert isinstance(image, ImageStack)

    with pytest.raises(ValueError):
        ImageStack.from_numpy_array(array.reshape((1, 1, 2, 2)))
        ImageStack.from_numpy_array(array.reshape((1, 2, 2)))
        ImageStack.from_numpy_array(array)
        ImageStack.from_numpy_array(array.reshape((1, 1, 1, 1, 2, 2)))


def test_from_numpy_array_automatically_handles_float_conversions():
    x = np.zeros((1, 1, 1, 20, 20), dtype=np.uint16)
    stack = ImageStack.from_numpy_array(x)
    assert stack.xarray.dtype == np.float32


def test_max_projection_preserves_dtype():
    original_dtype = np.float32
    array = np.ones((2, 2, 2), dtype=original_dtype)
    image = ImageStack.from_numpy_array(array.reshape((1, 1, 2, 2, 2)))

    max_projection = image.max_proj(Axes.CH, Axes.ROUND, Axes.ZPLANE)
    assert max_projection.xarray.dtype == original_dtype


def test_synthetic_spot_creation_raises_error_with_coords_too_small(synthetic_intensity_table):
    num_z = 0
    height = 40
    width = 50
    with pytest.raises(ValueError):
        ImageStack.synthetic_spots(synthetic_intensity_table, num_z, height, width)


def test_synthetic_spot_creation_produces_an_imagestack(synthetic_intensity_table):
    num_z = 12
    height = 50
    width = 40
    image = ImageStack.synthetic_spots(synthetic_intensity_table, num_z, height, width)
    assert isinstance(image, ImageStack)


def test_synthetic_spot_creation_produces_an_imagestack_with_correct_spot_location(
        synthetic_spot_pass_through_stack):

    codebook, true_intensities, image = synthetic_spot_pass_through_stack

    g, c, r = np.where(true_intensities.values)

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


# TODO ambrosejcarr: improve the tests here.
def test_imagestack_to_intensity_table():
    codebook, intensity_table, image = codebook_intensities_image_for_single_synthetic_spot()
    pixel_intensities = IntensityTable.from_image_stack(image)
    pixel_intensities = codebook.metric_decode(
        pixel_intensities, max_distance=0, min_intensity=1000, norm_order=2)
    assert isinstance(pixel_intensities, IntensityTable)


def test_imagestack_to_intensity_table_no_noise(synthetic_spot_pass_through_stack):
    codebook, intensity_table, image = synthetic_spot_pass_through_stack
    pixel_intensities = IntensityTable.from_image_stack(image)
    pixel_intensities = codebook.metric_decode(
        pixel_intensities, max_distance=0, min_intensity=1000, norm_order=2)
    assert isinstance(pixel_intensities, IntensityTable)


@pytest.mark.parametrize("format,count", (
    (ImageFormat.TIFF, 6),
    (ImageFormat.NUMPY, 6),
))
def test_imagestack_export(tmpdir, format, count, recwarn):
    """
    Save a synthetic stack to files and check the results
    """
    stack_shape = OrderedDict([(Axes.ROUND, 3), (Axes.CH, 2),
                               (Axes.ZPLANE, 1), (Axes.Y, 50), (Axes.X, 40)])

    physical_coords = OrderedDict([(PhysicalCoordinateTypes.X_MIN, X_COORDS[0]),
                                   (PhysicalCoordinateTypes.X_MAX, X_COORDS[1]),
                                   (PhysicalCoordinateTypes.Y_MIN, Y_COORDS[0]),
                                   (PhysicalCoordinateTypes.Y_MAX, Y_COORDS[1]),
                                   (PhysicalCoordinateTypes.Z_MIN, Z_COORDS[0]),
                                   (PhysicalCoordinateTypes.Z_MAX, Z_COORDS[1])])

    stack = test_utils.imagestack_with_coords_factory(stack_shape, physical_coords)

    stack_json = tmpdir / "output.json"
    stack.export(str(stack_json), tile_format=format)
    files = list([x for x in tmpdir.listdir() if str(x).endswith(format.file_ext)])
    loaded_stack = ImageStack.from_path_or_url(str(stack_json))
    verify_physical_coordinates(
        loaded_stack,
        X_COORDS,
        Y_COORDS,
        get_physical_coordinates_of_z_plane(Z_COORDS),
    )
    assert count == len(files)
    with open(files[0], "rb") as fh:
        format.reader_func(fh)
