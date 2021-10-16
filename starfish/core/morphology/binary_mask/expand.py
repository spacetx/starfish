from typing import MutableSequence, Tuple

import numpy as np


def fill_from_mask(
        mask: np.ndarray,
        offsets: Tuple[int, ...],
        fill_value: int,
        result_array: np.ndarray,
):
    """Take a binary mask and write `fill_value` to an array `result_array` where the binary mask
    has a True value.  The binary mask can have a different origin than the output array.  The
    relative offsets between the binary mask's origin an the output array's origin must be provided.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([True, True, False], dtype=bool)
    >>> mask
    array([ True,  True, False])
    >>> result_array = np.zeros(shape=(4,), dtype=np.uint32)
    >>> fill_from_mask(mask, (1,), 2, result_array)
    >>> result_array
    array([0, 2, 2, 0], dtype=uint32)
    """
    if mask.ndim != result_array.ndim:
        raise ValueError(
            f"mask ({mask.ndim}) should have the same number of dimensions as the result array "
            f"({result_array.ndim})")
    if mask.ndim != len(offsets):
        raise ValueError(
            f"mask ({mask.ndim}) should have the same number of dimensions as the number of "
            f"offsets ({len(offsets)})")

    _selector: MutableSequence[slice] = []
    for ix, (mask_size, result_size, axis_offset) in enumerate(
            zip(mask.shape, result_array.shape, offsets)):
        end_offset = axis_offset + mask_size
        if axis_offset < 0:
            raise ValueError(
                "{ix}th dimension has a negative offset ({axis_offset}), which is not permitted")
        if end_offset > result_size:
            raise ValueError(
                f"{ix}th dimension of mask does not fit within the result (ends at {end_offset}, "
                f"result size is {result_size}")

        _selector.append(slice(axis_offset, end_offset))
    selector: Tuple[slice, ...] = tuple(_selector)

    fill_value_array = result_array[selector]
    fill_value_array[mask] = fill_value
    result_array[selector] = fill_value_array
