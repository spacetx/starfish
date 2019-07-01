import numpy as np

from starfish.core.types import Axes
from .factories import synthetic_stack


def test_get_slice_simple_index():
    """
    Retrieve a slice across one of the axes at the end.  For instance, if the axes are
    (P, Q0,..., Qn-1, R), slice across either P or R.  This test has expectations regarding the
    ordering of the axes in the ImageStack.
    """
    stack = synthetic_stack()
    round_label = 1
    imageslice, axes = stack.get_slice(
        {Axes.ROUND: round_label}
    )
    assert axes == [Axes.CH, Axes.ZPLANE]

    y, x = stack.tile_shape

    for ch_label in range(stack.shape[Axes.CH]):
        for zplane_label in range(stack.shape[Axes.ZPLANE]):
            data = np.empty((y, x))
            data.fill(
                (round_label * stack.shape[Axes.CH] + ch_label) * stack.shape[Axes.ZPLANE]
                + zplane_label)

            assert data.all() == imageslice[ch_label, zplane_label].all()


def test_get_slice_middle_index():
    """
    Retrieve a slice across one of the axes in the middle.  For instance, if the axes are
    (P, Q0,..., Qn-1, R), slice across one of the Q axes.  This test has expectations regarding the
    ordering of the axes in the ImageStack.
    """
    stack = synthetic_stack()
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
    stack = synthetic_stack()
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
    stack = synthetic_stack()
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
    stack = synthetic_stack()
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
    stack = synthetic_stack()
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
    stack = synthetic_stack()
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
