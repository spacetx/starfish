import numpy as np
from scipy.ndimage.filters import gaussian_filter

from starfish import ImageStack
from starfish.core.spots.DecodeSpots.trace_builders import build_traces_nearest_neighbors
from starfish.core.spots.FindSpots import BlobDetector
from starfish.core.types import Axes


def traversing_code() -> ImageStack:
    """this code walks in a sequential direction, and should only be detectable from some anchors"""
    img = np.zeros((3, 2, 20, 50, 50), dtype=np.float32)

    # code 1
    img[0, 0, 5, 35, 35] = 10
    img[1, 1, 5, 32, 32] = 10
    img[2, 0, 5, 29, 29] = 10

    # blur points
    gaussian_filter(img, (0, 0, 0.5, 1.5, 1.5), output=img)

    return ImageStack.from_numpy(img)


def multiple_possible_neighbors() -> ImageStack:
    """this image is intended to be tested with anchor_round in {0, 1}, last round has more spots"""
    img = np.zeros((3, 2, 20, 50, 50), dtype=np.float32)

    # round 1
    img[0, 0, 5, 20, 40] = 10
    img[0, 0, 5, 40, 20] = 10

    # round 2
    img[1, 1, 5, 20, 40] = 10
    img[1, 1, 5, 40, 20] = 10

    # round 3
    img[2, 0, 5, 20, 40] = 10
    img[2, 0, 5, 35, 35] = 10
    img[2, 0, 5, 40, 20] = 10

    # blur points
    gaussian_filter(img, (0, 0, 0.5, 1.5, 1.5), output=img)

    return ImageStack.from_numpy(img)


def jitter_code() -> ImageStack:
    """this code has some minor jitter <= 3px at the most distant point"""
    img = np.zeros((3, 2, 20, 50, 50), dtype=np.float32)

    # code 1
    img[0, 0, 5, 35, 35] = 10
    img[1, 1, 5, 34, 35] = 10
    img[2, 0, 6, 35, 33] = 10

    # blur points
    gaussian_filter(img, (0, 0, 0.5, 1.5, 1.5), output=img)

    return ImageStack.from_numpy(img)


def two_perfect_codes() -> ImageStack:
    """this code has no jitter"""
    img = np.zeros((3, 2, 20, 50, 50), dtype=np.float32)

    # code 1
    img[0, 0, 5, 20, 35] = 10
    img[1, 1, 5, 20, 35] = 10
    img[2, 0, 5, 20, 35] = 10

    # code 1
    img[0, 0, 5, 40, 45] = 10
    img[1, 1, 5, 40, 45] = 10
    img[2, 0, 5, 40, 45] = 10

    # blur points
    gaussian_filter(img, (0, 0, 0.5, 1.5, 1.5), output=img)

    return ImageStack.from_numpy(img)


def blob_detector():
    return BlobDetector(min_sigma=1, max_sigma=4, num_sigma=30, threshold=.1)


def test_local_search_blob_detector_two_codes():
    stack = two_perfect_codes()
    bd = blob_detector()
    spot_results = bd.run(stack)

    intensity_table = build_traces_nearest_neighbors(spot_results=spot_results, anchor_round=1,
                                                     search_radius=1)

    # Sort features to ensure test stability
    sorted_intensity_table = intensity_table.sortby(["x", "y", "z"])
    assert sorted_intensity_table.shape == (2, 2, 3)
    assert np.all(sorted_intensity_table[1][Axes.X.value] == 45)
    assert np.all(sorted_intensity_table[1][Axes.Y.value] == 40)
    assert np.all(sorted_intensity_table[1][Axes.ZPLANE.value] == 5)


def test_local_search_blob_detector_jitter_code():
    stack = jitter_code()

    bd = blob_detector()
    spot_results = bd.run(stack)
    intensity_table = build_traces_nearest_neighbors(spot_results=spot_results, anchor_round=1,
                                                     search_radius=3)

    assert intensity_table.shape == (1, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0, 0]))
    assert np.all(c == np.array([0, 0, 1]))
    assert np.all(r == np.array([0, 2, 1]))

    # test again with smaller search radius
    bd = BlobDetector(min_sigma=1, max_sigma=4, num_sigma=30, threshold=.1)
    per_tile_spot_results = bd.run(stack)

    intensity_table = build_traces_nearest_neighbors(spot_results=per_tile_spot_results,
                                                     anchor_round=0,
                                                     search_radius=1)

    assert intensity_table.shape == (1, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == 0)
    assert np.all(c == 0)
    assert np.all(r == 0)


def test_local_search_blob_detector_traversing_code():
    stack = traversing_code()

    bd = blob_detector()
    spot_results = bd.run(stack)
    intensity_table = build_traces_nearest_neighbors(spot_results=spot_results, anchor_round=0,
                                                     search_radius=5)

    assert intensity_table.shape == (1, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0]))
    assert np.all(c == np.array([0, 1]))
    assert np.all(r == np.array([0, 1]))

    bd = blob_detector()
    spot_results = bd.run(stack)
    intensity_table = build_traces_nearest_neighbors(spot_results=spot_results, anchor_round=1,
                                                     search_radius=5)

    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0, 0]))
    assert np.all(c == np.array([0, 0, 1]))
    assert np.all(r == np.array([0, 2, 1]))


def test_local_search_blob_detector_multiple_neighbors():
    stack = multiple_possible_neighbors()

    bd = blob_detector()
    spot_results = bd.run(stack)
    intensity_table = build_traces_nearest_neighbors(spot_results=spot_results, anchor_round=0,
                                                     search_radius=4)

    # Sort features to ensure test stability
    sorted_intensity_table = intensity_table.sortby(["x", "y", "z"])
    assert sorted_intensity_table.shape == (2, 2, 3)
    assert np.all(sorted_intensity_table[Axes.ZPLANE.value] == (5, 5))
    assert np.all(sorted_intensity_table[Axes.Y.value] == (40, 20))
    assert np.all(sorted_intensity_table[Axes.X.value] == (20, 40))
