import numpy as np

from starfish.core.morphology.binary_mask.test.factories import binary_mask_collection_2d
from ..reduce import Reduce


def test_reduce():
    def make_initial(shape):
        constant_initial = np.zeros(shape=shape, dtype=np.bool)
        constant_initial[0, 0] = 1
        return constant_initial

    input_mask_collection = binary_mask_collection_2d()
    filt = Reduce("logical_xor", make_initial)
    output_mask_collection = filt.run(input_mask_collection)

    assert len(output_mask_collection) == 1
    uncropped_output = output_mask_collection.uncropped_mask(0)
    assert np.array_equal(
        np.asarray(uncropped_output),
        np.array(
            [[False, True, True, True, True, True],
             [False, False, False, False, False, False],
             [False, False, False, False, False, False],
             [False, False, False, True, True, True],
             [False, False, False, True, True, False],
             ],
            dtype=np.bool,
        ))
