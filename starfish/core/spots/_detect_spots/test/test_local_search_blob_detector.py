import numpy as np
from scipy.ndimage.filters import gaussian_filter

from starfish import ImageStack
from starfish.core.spots._detect_spots.local_search_blob_detector import LocalSearchBlobDetector
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


def local_search_blob_detector(search_radius: int, anchor_channel=0) -> LocalSearchBlobDetector:
    return LocalSearchBlobDetector(
        min_sigma=(0.4, 1.2, 1.2),
        max_sigma=(0.6, 1.7, 1.7),
        num_sigma=3,
        threshold=0.1,
        overlap=0.5,
        search_radius=search_radius,
        anchor_round=anchor_channel,
    )


def test_local_search_blob_detector_two_codes():
    stack = two_perfect_codes()
    lsbd = local_search_blob_detector(search_radius=1)
    intensity_table = lsbd.run(stack, n_processes=1)

    assert intensity_table.shape == (2, 2, 3)
    assert np.all(intensity_table[0][Axes.X.value] == 45)
    assert np.all(intensity_table[0][Axes.Y.value] == 40)
    assert np.all(intensity_table[0][Axes.ZPLANE.value] == 5)


def test_local_search_blob_detector_jitter_code():
    stack = jitter_code()
    lsbd = local_search_blob_detector(search_radius=3)
    intensity_table = lsbd.run(stack, n_processes=1)

    assert intensity_table.shape == (1, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0, 0]))
    assert np.all(c == np.array([0, 0, 1]))
    assert np.all(r == np.array([0, 2, 1]))

    # test again with smaller search radius
    lsbd = local_search_blob_detector(search_radius=1)
    intensity_table = lsbd.run(stack, n_processes=1)
    assert intensity_table.shape == (1, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0]))
    assert np.all(c == np.array([0]))
    assert np.all(r == np.array([0]))


def test_local_search_blob_detector_traversing_code():
    stack = traversing_code()
    lsbd = local_search_blob_detector(search_radius=5, anchor_channel=0)
    intensity_table = lsbd.run(stack, n_processes=1)

    assert intensity_table.shape == (1, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0]))
    assert np.all(c == np.array([0, 1]))
    assert np.all(r == np.array([0, 1]))

    lsbd = local_search_blob_detector(search_radius=5, anchor_channel=1)
    intensity_table = lsbd.run(stack, n_processes=1)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0, 0]))
    assert np.all(c == np.array([0, 0, 1]))
    assert np.all(r == np.array([0, 2, 1]))


def test_local_search_blob_detector_multiple_neighbors():
    stack = multiple_possible_neighbors()
    lsbd = local_search_blob_detector(search_radius=4, anchor_channel=0)
    intensity_table = lsbd.run(stack, n_processes=1)

    assert intensity_table.shape == (2, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(intensity_table[Axes.ZPLANE.value] == (5, 5))
    assert np.all(intensity_table[Axes.Y.value] == (40, 20))
    assert np.all(intensity_table[Axes.X.value] == (20, 40))
