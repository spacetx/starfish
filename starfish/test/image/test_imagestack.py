import numpy as np

from starfish.constants import Indices
from starfish.test.dataset_fixtures import synthetic_stack


def test_get_slice_simple_index():
    """
    Retrieve a slice across one of the indices at the end.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), slice across either P or R.
    """
    stack = synthetic_stack()
    hyb = 1
    imageslice, axes = stack.get_slice(
        {Indices.HYB: hyb}
    )
    assert axes == [Indices.CH, Indices.Z]

    Y, X = stack.tile_shape

    for ch in range(stack.shape[Indices.CH]):
        for z in range(stack.shape[Indices.Z]):
            data = np.empty((Y, X))
            data.fill((hyb * stack.shape[Indices.CH] + ch) * stack.shape[Indices.Z] + z)

            assert data.all() == imageslice[ch, z].all()


def test_get_slice_middle_index():
    """
    Retrieve a slice across one of the indices in the middle.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), slice across one of the Q axes.
    """
    stack = synthetic_stack()
    ch = 1
    imageslice, axes = stack.get_slice(
        {Indices.CH: ch}
    )
    assert axes == [Indices.HYB, Indices.Z]

    Y, X = stack.tile_shape

    for hyb in range(stack.shape[Indices.HYB]):
        for z in range(stack.shape[Indices.Z]):
            data = np.empty((Y, X))
            data.fill((hyb * stack.shape[Indices.CH] + ch) * stack.shape[Indices.Z] + z)

            assert data.all() == imageslice[hyb, z].all()


def test_get_slice_range():
    """
    Retrieve a slice across a range of one of the dimensions.
    """
    stack = synthetic_stack()
    zrange = slice(1, 3)
    imageslice, axes = stack.get_slice(
        {Indices.Z: zrange}
    )
    Y, X = stack.tile_shape
    assert axes == [Indices.HYB, Indices.CH, Indices.Z]

    for hyb in range(stack.shape[Indices.HYB]):
        for ch in range(stack.shape[Indices.CH]):
            for z in range(zrange.stop - zrange.start):
                data = np.empty((Y, X))
                data.fill((hyb * stack.shape[Indices.CH] + ch) * stack.shape[Indices.Z] +
                          (z + zrange.start))

                assert data.all() == imageslice[hyb, ch, z].all()


def test_set_slice_simple_index():
    """
    Sets a slice across one of the indices at the end.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), sets a slice across either P or R.
    """
    stack = synthetic_stack()
    hyb = 1
    Y, X = stack.tile_shape

    expected = np.ones((stack.shape[Indices.CH], stack.shape[Indices.Z], Y, X)) * 2
    index = {Indices.HYB: hyb}

    stack.set_slice(index, expected)

    assert np.array_equal(stack.get_slice(index)[0], expected)


def test_set_slice_middle_index():
    """
    Sets a slice across one of the indices in the middle.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), slice across one of the Q axes.
    """
    stack = synthetic_stack()
    ch = 1
    Y, X = stack.tile_shape

    expected = np.ones((stack.shape[Indices.HYB], stack.shape[Indices.Z], Y, X)) * 2
    index = {Indices.CH: ch}

    stack.set_slice(index, expected)

    assert np.array_equal(stack.get_slice(index)[0], expected)


def test_set_slice_range():
    """
    Sets a slice across a range of one of the dimensions.
    """
    stack = synthetic_stack()
    zrange = slice(1, 3)
    Y, X = stack.tile_shape

    expected = np.ones((
        stack.shape[Indices.HYB],
        stack.shape[Indices.CH],
        zrange.stop - zrange.start,
        Y, X)) * 10
    index = {Indices.Z: zrange}

    stack.set_slice(index, expected)

    assert np.array_equal(stack.get_slice(index)[0], expected)
