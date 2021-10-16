from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from starfish.core.types import ArrayLike, Axes, Coordinates, Number
from ..binary_mask import BinaryMaskCollection


def label_array_2d() -> Tuple[np.ndarray, Mapping[Coordinates, ArrayLike[Number]]]:
    """Convenience method to return a 2D label array with corresponding physical coordinates."""
    label_array = np.zeros((5, 6), dtype=np.int32)
    label_array[0] = 1
    label_array[3:5, 3:6] = 2
    label_array[-1, -1] = 0

    physical_ticks = {
        Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
        Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12, 13.2]}

    return label_array, physical_ticks


def binary_arrays_2d() -> Tuple[Sequence[np.ndarray], Mapping[Coordinates, ArrayLike[Number]]]:
    """Convenience method to return a set of 2D binary arrays with corresponding physical
    coordinates."""
    binary_arrays = [
        np.zeros((5, 6), dtype=bool),
        np.zeros((5, 6), dtype=bool),
    ]
    binary_arrays[0][0] = True
    binary_arrays[1][3:5, 3:6] = True
    binary_arrays[1][-1, -1] = False

    physical_ticks = {
        Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
        Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12, 13.2]}

    return binary_arrays, physical_ticks


def label_array_3d() -> Tuple[np.ndarray, Mapping[Coordinates, ArrayLike[Number]]]:
    """Convenience method to return a 3D label array with corresponding physical coordinates."""
    label_array = np.zeros((2, 5, 6), dtype=np.int32)
    label_array[0, 0] = 1
    label_array[:, 3:5, 3:6] = 2
    label_array[-1, -1, -1] = 0

    physical_ticks = {
        Coordinates.Z: [0.0, 1.0],
        Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
        Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12, 13.2]
    }

    return label_array, physical_ticks


def binary_arrays_3d() -> Tuple[Sequence[np.ndarray], Mapping[Coordinates, ArrayLike[Number]]]:
    """Convenience method to return a set of 2D binary arrays with corresponding physical
    coordinates."""
    binary_arrays = [
        np.zeros((2, 5, 6), dtype=bool),
        np.zeros((2, 5, 6), dtype=bool),
    ]
    binary_arrays[0][0, 0] = True
    binary_arrays[1][:, 3:5, 3:6] = True
    binary_arrays[1][-1, -1, -1] = False

    physical_ticks = {
        Coordinates.Z: [0.0, 1.0],
        Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
        Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12, 13.2]
    }

    return binary_arrays, physical_ticks


def binary_mask_collection_2d(
        label_array: Optional[np.ndarray] = None,
        physical_ticks: Optional[Mapping[Coordinates, ArrayLike[Number]]] = None,
        pixel_ticks: Optional[Mapping[Axes, ArrayLike[int]]] = None):
    """Convenience method to return a 2D binary mask collection."""
    if label_array is None or physical_ticks is None:
        new_label_array, new_physical_ticks = label_array_2d()
        label_array = label_array or new_label_array
        physical_ticks = physical_ticks or new_physical_ticks

    return BinaryMaskCollection.from_label_array_and_ticks(
        label_array,
        pixel_ticks,
        physical_ticks,
        None
    )


def binary_mask_collection_3d(
        label_array: Optional[np.ndarray] = None,
        physical_ticks: Optional[Mapping[Coordinates, ArrayLike[Number]]] = None,
        pixel_ticks: Optional[Mapping[Axes, ArrayLike[int]]] = None):
    """Convenience method to return a 3D binary mask collection."""
    if label_array is None or physical_ticks is None:
        new_label_array, new_physical_ticks = label_array_3d()
        label_array = label_array or new_label_array
        physical_ticks = physical_ticks or new_physical_ticks

    return BinaryMaskCollection.from_label_array_and_ticks(
        label_array,
        pixel_ticks,
        physical_ticks,
        None
    )
