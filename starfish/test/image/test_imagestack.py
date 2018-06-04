from copy import deepcopy

import numpy as np

from starfish.constants import Indices
from starfish.test.dataset_fixtures import synthetic_stack


def test_get_slice_simple_index(synthetic_stack):
    """
    Retrieve a slice across one of the indices at the end.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), slice across either P or R.
    """
    hyb = 1
    imageslice, axes = synthetic_stack.get_slice(
        {Indices.HYB: hyb}
    )
    assert axes == [Indices.CH, Indices.Z]

    Y, X = synthetic_stack.tile_shape

    for ch in range(synthetic_stack.shape[Indices.CH]):
        for z in range(synthetic_stack.shape[Indices.Z]):
            data = np.empty((Y, X))
            data.fill((hyb * synthetic_stack.shape[Indices.CH] + ch) * synthetic_stack.shape[Indices.Z] + z)

            assert data.all() == imageslice[ch, z].all()


def test_get_slice_middle_index(synthetic_stack):
    """
    Retrieve a slice across one of the indices in the middle.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), slice across one of the Q axes.
    """
    ch = 1
    imageslice, axes = synthetic_stack.get_slice(
        {Indices.CH: ch}
    )
    assert axes == [Indices.HYB, Indices.Z]

    Y, X = synthetic_stack.tile_shape

    for hyb in range(synthetic_stack.shape[Indices.HYB]):
        for z in range(synthetic_stack.shape[Indices.Z]):
            data = np.empty((Y, X))
            data.fill((hyb * synthetic_stack.shape[Indices.CH] + ch) * synthetic_stack.shape[Indices.Z] + z)

            assert data.all() == imageslice[hyb, z].all()


def test_get_slice_range(synthetic_stack):
    """
    Retrieve a slice across a range of one of the dimensions.
    """
    zrange = slice(1, 3)
    imageslice, axes = synthetic_stack.get_slice(
        {Indices.Z: zrange}
    )
    Y, X = synthetic_stack.tile_shape
    assert axes == [Indices.HYB, Indices.CH, Indices.Z]

    for hyb in range(synthetic_stack.shape[Indices.HYB]):
        for ch in range(synthetic_stack.shape[Indices.CH]):
            for z in range(zrange.stop - zrange.start):
                data = np.empty((Y, X))
                data.fill((hyb * synthetic_stack.shape[Indices.CH] + ch) * synthetic_stack.shape[Indices.Z] +
                          (z + zrange.start))

                assert data.all() == imageslice[hyb, ch, z].all()


def test_set_slice_simple_index(synthetic_stack):
    """
    Sets a slice across one of the indices at the end.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), sets a slice across either P or R.
    """
    synthetic_stack = deepcopy(synthetic_stack)
    hyb = 1
    Y, X = synthetic_stack.tile_shape

    expected = np.ones((synthetic_stack.shape[Indices.CH], synthetic_stack.shape[Indices.Z], Y, X)) * 2
    index = {Indices.HYB: hyb}

    synthetic_stack.set_slice(index, expected)

    assert np.array_equal(synthetic_stack.get_slice(index)[0], expected)


def test_set_slice_middle_index(synthetic_stack):
    """
    Sets a slice across one of the indices in the middle.  For instance, if the dimensions are
    (P, Q0,..., Qn-1, R), slice across one of the Q axes.
    """
    synthetic_stack = deepcopy(synthetic_stack)
    ch = 1
    Y, X = synthetic_stack.tile_shape

    expected = np.ones((synthetic_stack.shape[Indices.HYB], synthetic_stack.shape[Indices.Z], Y, X)) * 2
    index = {Indices.CH: ch}

    synthetic_stack.set_slice(index, expected)

    assert np.array_equal(synthetic_stack.get_slice(index)[0], expected)


def test_set_slice_range(synthetic_stack):
    """
    Sets a slice across a range of one of the dimensions.
    """
    synthetic_stack = deepcopy(synthetic_stack)
    zrange = slice(1, 3)
    Y, X = synthetic_stack.tile_shape

    expected = np.ones((
        synthetic_stack.shape[Indices.HYB],
        synthetic_stack.shape[Indices.CH],
        zrange.stop - zrange.start,
        Y, X)) * 10
    index = {Indices.Z: zrange}

    synthetic_stack.set_slice(index, expected)

    assert np.array_equal(synthetic_stack.get_slice(index)[0], expected)
