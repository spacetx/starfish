import numpy as np
import pytest

from starfish.constants import Indices
from starfish.image import ImageStack
# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import (
    synthetic_intensity_table, synthetic_spot_pass_through_stack, loaded_codebook,
    synthetic_dataset_with_truth_values, simple_codebook_json, simple_codebook_array)


def test_get_slice_simple_index():
    """
    Retrieve a slice across one of the indices at the end.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), slice across either P or R.
    """
    stack = ImageStack.synthetic_stack()
    hyb = 1
    imageslice, axes = stack.get_slice(
        {Indices.HYB: hyb}
    )
    assert axes == [Indices.CH, Indices.Z]

    y, x = stack.tile_shape

    for ch in range(stack.shape[Indices.CH]):
        for z in range(stack.shape[Indices.Z]):
            data = np.empty((y, x))
            data.fill((hyb * stack.shape[Indices.CH] + ch) * stack.shape[Indices.Z] + z)

            assert data.all() == imageslice[ch, z].all()


def test_get_slice_middle_index():
    """
    Retrieve a slice across one of the indices in the middle.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), slice across one of the Q axes.
    """
    stack = ImageStack.synthetic_stack()
    ch = 1
    imageslice, axes = stack.get_slice(
        {Indices.CH: ch}
    )
    assert axes == [Indices.HYB, Indices.Z]

    y, x = stack.tile_shape

    for hyb in range(stack.shape[Indices.HYB]):
        for z in range(stack.shape[Indices.Z]):
            data = np.empty((y, x))
            data.fill((hyb * stack.shape[Indices.CH] + ch) * stack.shape[Indices.Z] + z)

            assert data.all() == imageslice[hyb, z].all()


def test_get_slice_range():
    """
    Retrieve a slice across a range of one of the dimensions.
    """
    stack = ImageStack.synthetic_stack()
    zrange = slice(1, 3)
    imageslice, axes = stack.get_slice(
        {Indices.Z: zrange}
    )
    y, x = stack.tile_shape
    assert axes == [Indices.HYB, Indices.CH, Indices.Z]

    for hyb in range(stack.shape[Indices.HYB]):
        for ch in range(stack.shape[Indices.CH]):
            for z in range(zrange.stop - zrange.start):
                data = np.empty((y, x))
                data.fill((hyb * stack.shape[Indices.CH] + ch) * stack.shape[Indices.Z] +
                          (z + zrange.start))

                assert data.all() == imageslice[hyb, ch, z].all()


def test_set_slice_simple_index():
    """
    Sets a slice across one of the indices at the end.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), sets a slice across either P or R.
    """
    stack = ImageStack.synthetic_stack()
    hyb = 1
    y, x = stack.tile_shape

    expected = np.ones((stack.shape[Indices.CH], stack.shape[Indices.Z], y, x)) * 2
    index = {Indices.HYB: hyb}

    stack.set_slice(index, expected)

    assert np.array_equal(stack.get_slice(index)[0], expected)


def test_set_slice_middle_index():
    """
    Sets a slice across one of the indices in the middle.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), slice across one of the Q axes.
    """
    stack = ImageStack.synthetic_stack()
    ch = 1
    y, x = stack.tile_shape

    expected = np.ones((stack.shape[Indices.HYB], stack.shape[Indices.Z], y, x)) * 2
    index = {Indices.CH: ch}

    stack.set_slice(index, expected)

    assert np.array_equal(stack.get_slice(index)[0], expected)


def test_set_slice_range():
    """
    Sets a slice across a range of one of the dimensions.
    """
    stack = ImageStack.synthetic_stack()
    zrange = slice(1, 3)
    y, x = stack.tile_shape

    expected = np.ones((
        stack.shape[Indices.HYB],
        stack.shape[Indices.CH],
        zrange.stop - zrange.start,
        y, x)) * 10
    index = {Indices.Z: zrange}

    stack.set_slice(index, expected)

    assert np.array_equal(stack.get_slice(index)[0], expected)


def test_from_numpy_array_preserves_data():
    array = np.random.random((1, 1, 1, 2, 2))
    image_stack = ImageStack.from_numpy_array(array)
    assert np.array_equal(array, image_stack.numpy_array)


def test_from_numpy_array_raises_error_when_incorrect_dims_passed():
    array = np.ones((2, 2))
    # verify this method works with the correct shape
    image = ImageStack.from_numpy_array(array.reshape((1, 1, 1, 2, 2)))
    assert isinstance(image, ImageStack)

    with pytest.raises(ValueError):
        ImageStack.from_numpy_array(array.reshape((1, 1, 2, 2)))
        ImageStack.from_numpy_array(array.reshape((1, 2, 2)))
        ImageStack.from_numpy_array(array)
        ImageStack.from_numpy_array(array.reshape((1, 1, 1, 1, 2, 2)))


def test_from_numpy_array_preserves_dtype():
    original_dtype = np.uint16
    array = np.ones((2, 2, 2), dtype=original_dtype)
    image = ImageStack.from_numpy_array(array.reshape((1, 1, 2, 2, 2)))
    assert image.numpy_array.dtype == original_dtype


def test_max_projection_preserves_dtype():
    original_dtype = np.uint16
    array = np.ones((2, 2, 2), dtype=original_dtype)
    image = ImageStack.from_numpy_array(array.reshape((1, 1, 2, 2, 2)))

    max_projection = image.max_proj(Indices.CH, Indices.HYB, Indices.Z)
    assert max_projection.dtype == original_dtype


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

    g, c, h = np.where(true_intensities.values)

    x = np.empty_like(g)
    y = np.empty_like(g)
    z = np.empty_like(g)
    breaks = np.concatenate([
        np.array([0]),
        np.where(np.diff(g))[0] + 1,
        np.array([g.shape[0]])
    ])
    for i in np.arange(len(breaks) - 1):
        x[breaks[i]: breaks[i + 1]] = true_intensities.coords['x'][i]
        y[breaks[i]: breaks[i + 1]] = true_intensities.coords['y'][i]
        z[breaks[i]: breaks[i + 1]] = true_intensities.coords['z'][i]

    # only 8 values should be set, since there are only 8 locations across the tensor
    assert np.sum(image.numpy_array != 0) == 8

    assert np.array_equal(
        image.numpy_array[h, c, z, y, x],
        true_intensities.values[np.where(true_intensities)])
