"""
These tests center around creating an ImageStack but selectively loading data from the original
TileSet.
"""
import numpy as np

from starfish.core.experiment.builder.test.factories.unique_tiles import unique_data
from starfish.core.types import Axes
from .factories.unique_tiles import (
    unique_tiles_imagestack, X_COORDS, Y_COORDS, Z_COORDS,
)
from .imagestack_test_utils import (
    recalculate_physical_coordinate_range,
    verify_physical_coordinates,
    verify_stack_data,
)
from ..imagestack import ImageStack
from ..parser.crop import CropParameters


NUM_FOV = 1
NUM_ROUND = 3
NUM_CH = 4
NUM_ZPLANE = 2

ROUND_LABELS = list(range(NUM_ROUND))
CH_LABELS = list(range(NUM_CH))
ZPLANE_LABELS = list(range(NUM_ZPLANE))
HEIGHT = 40
WIDTH = 60


def expected_data(round_label: int, ch_label: int, zplane_label: int):
    return unique_data(
        0, round_label, ch_label, zplane_label,
        NUM_FOV, NUM_ROUND, NUM_CH, NUM_ZPLANE,
        HEIGHT, WIDTH
    )


def setup_imagestack(crop_parameters: CropParameters) -> ImageStack:
    return unique_tiles_imagestack(
        ROUND_LABELS, CH_LABELS, ZPLANE_LABELS, HEIGHT, WIDTH, crop_parameters)


def test_crop_rcz():
    """Build an imagestack that contains a crop in r/c/z.  Verify that the appropriate tiles are
    loaded.
    """
    rounds = [1]
    chs = [2, 3]

    crop_parameters = CropParameters(
        permitted_rounds=rounds,
        permitted_chs=chs,
    )
    stack = setup_imagestack(crop_parameters)

    assert stack.axis_labels(Axes.ROUND) == rounds
    assert stack.axis_labels(Axes.CH) == chs
    assert stack.axis_labels(Axes.ZPLANE) == ZPLANE_LABELS

    expected_zplane_coordinates = np.linspace(Z_COORDS[0], Z_COORDS[1], NUM_ZPLANE)

    for zplane_label in stack.axis_labels(Axes.ZPLANE):
        for round_label in stack.axis_labels(Axes.ROUND):
            for ch_label in stack.axis_labels(Axes.CH):
                expected_tile_data = expected_data(round_label, ch_label, zplane_label)

                verify_stack_data(
                    stack,
                    {Axes.ROUND: round_label, Axes.CH: ch_label, Axes.ZPLANE: zplane_label},
                    expected_tile_data,
                )

        zplane_index = ZPLANE_LABELS.index(zplane_label)
        expected_zplane_coordinate = expected_zplane_coordinates[zplane_index]
        verify_physical_coordinates(
            stack,
            X_COORDS,
            Y_COORDS,
            expected_zplane_coordinate,
            zplane_label,
        )


def test_crop_xy():
    """Build an imagestack that contains a crop in x/y.  Verify that the data is sliced correctly.
    """
    X_SLICE = (10, 30)
    Y_SLICE = (15, 40)
    crop_parameters = CropParameters(
        x_slice=slice(*X_SLICE),
        y_slice=slice(*Y_SLICE),
    )
    stack = setup_imagestack(crop_parameters)

    assert stack.axis_labels(Axes.ROUND) == ROUND_LABELS
    assert stack.axis_labels(Axes.CH) == CH_LABELS
    assert stack.axis_labels(Axes.ZPLANE) == ZPLANE_LABELS

    assert stack.raw_shape[3] == Y_SLICE[1] - Y_SLICE[0]
    assert stack.raw_shape[4] == X_SLICE[1] - X_SLICE[0]

    expected_zplane_coordinates = np.linspace(Z_COORDS[0], Z_COORDS[1], NUM_ZPLANE)

    for zplane in stack.axis_labels(Axes.ZPLANE):
        for round_ in stack.axis_labels(Axes.ROUND):
            for ch in stack.axis_labels(Axes.CH):
                expected_tile_data = expected_data(round_, ch, zplane)
                expected_tile_data = expected_tile_data[
                    Y_SLICE[0]:Y_SLICE[1], X_SLICE[0]:X_SLICE[1]]

                verify_stack_data(
                    stack,
                    {Axes.ROUND: round_, Axes.CH: ch, Axes.ZPLANE: zplane},
                    expected_tile_data,
                )

        # the coordinates should be rescaled.  verify that the coordinates on the ImageStack
        # are also rescaled.
        original_x_coordinates = X_COORDS
        expected_x_coordinates = recalculate_physical_coordinate_range(
            original_x_coordinates[0], original_x_coordinates[1],
            WIDTH,
            slice(*X_SLICE),
        )

        original_y_coordinates = Y_COORDS
        expected_y_coordinates = recalculate_physical_coordinate_range(
            original_y_coordinates[0], original_y_coordinates[1],
            HEIGHT,
            slice(*Y_SLICE),
        )

        zplane_index = ZPLANE_LABELS.index(zplane)
        expected_zplane_coordinate = expected_zplane_coordinates[zplane_index]

        verify_physical_coordinates(
            stack,
            expected_x_coordinates,
            expected_y_coordinates,
            expected_zplane_coordinate,
            zplane,
        )
