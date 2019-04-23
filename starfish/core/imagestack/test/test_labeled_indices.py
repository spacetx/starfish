"""
These tests center around creating an ImageStack with labeled indices and verifying that operations
on such an ImageStack work.
"""
import numpy as np

from starfish.types import Axes
from .factories.unique_tiles import (
    unique_data, unique_tiles_imagestack, X_COORDS, Y_COORDS, Z_COORDS,
)
from .imagestack_test_utils import verify_physical_coordinates, verify_stack_data
from ..imagestack import ImageStack
from ..physical_coordinate_calculator import get_physical_coordinates_of_z_plane

ROUND_LABELS = (1, 4, 6)
CH_LABELS = (2, 4, 6, 8)
ZPLANE_LABELS = (3, 4)
HEIGHT = 2
WIDTH = 4

NUM_ROUND = len(ROUND_LABELS)
NUM_CH = len(CH_LABELS)
NUM_ZPLANE = len(ZPLANE_LABELS)


def expected_data(round_: int, ch: int, zplane: int):
    return unique_data(round_, ch, zplane, NUM_ROUND, NUM_CH, NUM_ZPLANE, HEIGHT, WIDTH)


def setup_imagestack() -> ImageStack:
    return unique_tiles_imagestack(
        ROUND_LABELS, CH_LABELS, ZPLANE_LABELS, HEIGHT, WIDTH)


def test_labeled_indices_read():
    """Build an imagestack with labeled indices (i.e., indices that do not start at 0 or are not
    sequential non-negative integers).  Verify that get_slice behaves correctly.
    """
    stack = setup_imagestack()

    for round_ in stack.axis_labels(Axes.ROUND):
        for ch in stack.axis_labels(Axes.CH):
            for zplane in stack.axis_labels(Axes.ZPLANE):
                verify_stack_data(
                    stack,
                    {Axes.ROUND: round_, Axes.CH: ch, Axes.ZPLANE: zplane},
                    expected_data(round_, ch, zplane),
                )


def test_labeled_indices_set_slice():
    """Build an imagestack with labeled indices (i.e., indices that do not start at 0 or are not
    sequential non-negative integers).  Verify that set_slice behaves correctly.
    """
    for round_ in ROUND_LABELS:
        for ch in CH_LABELS:
            for zplane in ZPLANE_LABELS:
                stack = setup_imagestack()
                zeros = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

                stack.set_slice(
                    {Axes.ROUND: round_, Axes.CH: ch, Axes.ZPLANE: zplane}, zeros)

                for selector in stack._iter_axes({Axes.ROUND, Axes.CH, Axes.ZPLANE}):
                    if (selector[Axes.ROUND] == round_
                            and selector[Axes.CH] == ch
                            and selector[Axes.ZPLANE] == zplane):
                        expected_fill_value = zeros
                    else:
                        expected_fill_value = expected_data(
                            selector[Axes.ROUND], selector[Axes.CH], selector[Axes.ZPLANE])

                    verify_stack_data(stack, selector, expected_fill_value)


def test_labeled_indices_sel_single_tile():
    """Select a single tile across each index from an ImageStack with labeled indices.  Verify that
    the data is correct and that the physical coordinates are correctly set."""
    stack = setup_imagestack()

    for selector in stack._iter_axes({Axes.ROUND, Axes.CH, Axes.ZPLANE}):
        subselected = stack.sel(selector)

        # verify that the subselected stack has the correct index labels.
        for index_type in (Axes.ROUND, Axes.CH, Axes.ZPLANE):
            assert subselected.axis_labels(index_type) == [selector[index_type]]

        # verify that the subselected stack has the correct data.
        expected_tile_data = expected_data(
            selector[Axes.ROUND], selector[Axes.CH], selector[Axes.ZPLANE])
        verify_stack_data(stack, selector, expected_tile_data)

        # assert that the physical coordinate values are what we expect.
    verify_physical_coordinates(
        stack,
        X_COORDS,
        Y_COORDS,
        get_physical_coordinates_of_z_plane(Z_COORDS),
    )


def test_labeled_indices_sel_slice():
    """Select a single tile across each index from an ImageStack with labeled indices.  Verify that
    the data is correct and that the physical coordinates are correctly set."""
    stack = setup_imagestack()
    selector = {Axes.ROUND: slice(None, 4), Axes.CH: slice(4, 6), Axes.ZPLANE: 4}
    subselected = stack.sel(selector)

    # verify that the subselected stack has the correct index labels.
    for index_type, expected_results in (
            (Axes.ROUND, [1, 4]),
            (Axes.CH, [4, 6]),
            (Axes.ZPLANE, [4],)):
        assert subselected.axis_labels(index_type) == expected_results

    for selectors in subselected._iter_axes({Axes.ROUND, Axes.CH, Axes.ZPLANE}):
        # verify that the subselected stack has the correct data.
        expected_tile_data = expected_data(
            selectors[Axes.ROUND], selectors[Axes.CH], selectors[Axes.ZPLANE])
        verify_stack_data(subselected, selectors, expected_tile_data)

        # verify that each tile in the subselected stack has the correct physical coordinates.
    verify_physical_coordinates(
        stack,
        X_COORDS,
        Y_COORDS,
        get_physical_coordinates_of_z_plane(Z_COORDS),
    )


def multiply(array, value):
    return array * value


def test_labeled_indices_apply():
    """Build an imagestack with labeled indices (i.e., indices that do not start at 0 or are not
    sequential non-negative integers).  Verify that apply behaves correctly.
    """
    stack = setup_imagestack()

    for round_ in stack.axis_labels(Axes.ROUND):
        for ch in stack.axis_labels(Axes.CH):
            for zplane in stack.axis_labels(Axes.ZPLANE):
                verify_stack_data(
                    stack,
                    {Axes.ROUND: round_, Axes.CH: ch, Axes.ZPLANE: zplane},
                    expected_data(round_, ch, zplane),
                )

    output = stack.apply(multiply, value=0.5)

    for round_ in stack.axis_labels(Axes.ROUND):
        for ch in stack.axis_labels(Axes.CH):
            for zplane in stack.axis_labels(Axes.ZPLANE):
                verify_stack_data(
                    output,
                    {Axes.ROUND: round_, Axes.CH: ch, Axes.ZPLANE: zplane},
                    expected_data(round_, ch, zplane) * 0.5,
                )
