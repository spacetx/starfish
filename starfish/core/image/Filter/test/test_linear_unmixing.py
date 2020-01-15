import numpy as np

from starfish import ImageStack
from starfish.core.image.Filter.linear_unmixing import LinearUnmixing
from starfish.core.types import Levels

def setup_linear_unmixing_test():
    """
        Create the image stack, coeff matrix, and reference result
        for the linear unmixing test

    """
    # Create image
    r, c, z, y, x = 1, 3, 6, 5, 4
    im = np.ones((r, c, z, y, x), dtype=np.float32)

    # Create a single pixel with zero intensity
    im[0, 0, 0, 0, 0] = 0
    stack = ImageStack.from_numpy(im)

    # Create coefficients matrix
    coeff_mat = np.array(
        [[ 1.00, -0.25,  0.00],  # noqa
         [-0.25,  1.00, -0.25],  # noqa
         [-0.10,  0.00,  1.00]]  # noqa
    )

    # Create reference result
    ref_result = im.copy()
    ref_result[:, 0, ...] = 0.65 * np.ones((z, y, x))
    ref_result[:, 1, ...] = 0.75 * np.ones((z, y, x))
    ref_result[:, 2, ...] = 0.75 * np.ones((z, y, x))

    # Account for the zero-value pixel
    ref_result[0, 0, 0, 0, 0] = 0
    ref_result[0, 1, 0, 0, 0] = 1

    return stack, coeff_mat, ref_result

def test_linear_unmixing():
    """ Test the linear unmixing filter """

    stack, coeff_mat, ref_result = setup_linear_unmixing_test()

    filter_unmix = LinearUnmixing(coeff_mat=coeff_mat, level_method=Levels.CLIP)
    stack2 = filter_unmix.run(stack, in_place=False, verbose=False)

    assert np.allclose(ref_result, np.asarray(stack2.xarray))
