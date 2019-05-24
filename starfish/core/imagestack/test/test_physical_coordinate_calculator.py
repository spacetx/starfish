import numpy as np

from .imagestack_test_utils import recalculate_physical_coordinate_range

COORDS = 0.01, 0.1


def test_physical_coordinate_calculator():
    # coords of the starting pixel should be (0.01, 0.01)
    start, end = recalculate_physical_coordinate_range(COORDS[0], COORDS[1], 60, 0)
    assert np.isclose(start, 0.01)
    assert np.isclose(end, 0.01)

    # coords of the last pixel should be (0.1, 0.1)
    start, end = recalculate_physical_coordinate_range(COORDS[0], COORDS[1], 60, 59)
    assert np.isclose(start, 0.1)
    assert np.isclose(end, 0.1)

    # the original range should be (0.01, 0.1)
    start, end = recalculate_physical_coordinate_range(COORDS[0], COORDS[1], 60, slice(0, 60))
    assert np.isclose(start, 0.01)
    assert np.isclose(end, 0.1)

    # another way of expressing the original range
    start, end = recalculate_physical_coordinate_range(COORDS[0], COORDS[1], 60, slice(0, -1))
    assert np.isclose(start, 0.01)
    assert np.isclose(end, 0.1)

    # yet another way of expressing the original range
    start, end = recalculate_physical_coordinate_range(COORDS[0], COORDS[1], 60, slice(0, None))
    assert np.isclose(start, 0.01)
    assert np.isclose(end, 0.1)

    # and yet another way of expressing the original range
    start, end = recalculate_physical_coordinate_range(COORDS[0], COORDS[1], 60, slice(None, -1))
    assert np.isclose(start, 0.01)
    assert np.isclose(end, 0.1)

    # coords of the first half should be (0.01, 0.0557627)
    start, end = recalculate_physical_coordinate_range(COORDS[0], COORDS[1], 60, slice(0, 30))
    assert np.isclose(start, 0.01)
    assert np.isclose(end, 0.054237288135593)

    # coords of the second half should be (0.01, 0.0557627)
    start, end = recalculate_physical_coordinate_range(COORDS[0], COORDS[1], 60, slice(30, 60))
    assert np.isclose(start, 0.055762711864407)
    assert np.isclose(end, 0.1)
