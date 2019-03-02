import numpy as np

from starfish import ImageStack
from starfish.image._filter.linear_unmixing import LinearUnmixing

def setup_linear_unmixing_test():
    """
        Create the image stack, coeff matrix, and reference result
        for the linear unmixing test

    """
    # Create image
    r, c, z, y, x = 2, 3, 6, 5, 4
    im = np.ones((r, c, z, y, x), dtype=np.float32)
    stack = ImageStack.from_numpy_array(im)

    # Create coefficients matrix
    coeff_mat = np.array([[1, 0, 0], [-0.25, 1, -0.25], [0, 0, 1]])

    # Create reference result
    ref_result = np.ones((r, c, z, y, x))
    ref_result[:, 1, ...] = 0.5 * np.ones((z, y, x))

    return stack, coeff_mat, ref_result

def test_linear_unmixing():
    """ Test the linear unmixing filter """

    stack, coeff_mat, ref_result = setup_linear_unmixing_test()

    filter_unmix = LinearUnmixing(coeff_mat=coeff_mat)
    stack2 = filter_unmix.run(stack, in_place=False, verbose=False, n_processes=1)

    assert np.all(ref_result == stack2.xarray.values)
