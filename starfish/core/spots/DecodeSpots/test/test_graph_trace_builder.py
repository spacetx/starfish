import numpy as np
from scipy.ndimage.filters import gaussian_filter

from starfish import ImageStack
from starfish.core.spots.DecodeSpots.trace_builders import build_traces_graph_based
from starfish.core.spots.FindSpots import HMax
from starfish.core.types import Axes


def traversing_code() -> ImageStack:
    """this code walks in a sequential direction"""
    img = np.zeros((3, 2, 20, 50, 50), dtype=np.float32)

    # code 1
    img[0, 0, 5, 35, 35] = 10
    img[1, 1, 5, 32, 32] = 10
    img[2, 0, 5, 29, 29] = 10

    # blur points
    gaussian_filter(img, (0, 0, 0.5, 1.5, 1.5), output=img)

    return ImageStack.from_numpy(img)


def empty_data() -> ImageStack:
    """this code walks in a sequential direction"""
    img = np.zeros((3, 2, 20, 50, 50), dtype=np.float32)

    return ImageStack.from_numpy(img)


def multiple_possible_neighbors() -> ImageStack:
    """last round has more spots"""
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


def multiple_possible_neighbors_with_jitter() -> ImageStack:
    """last round has more spots and spots have some jitter <= 10px"""
    img = np.zeros((3, 2, 20, 50, 50), dtype=np.float32)

    # round 1
    img[0, 0, 5, 20, 40] = 10
    img[0, 0, 5, 40, 10] = 10

    # round 2
    img[1, 1, 5, 20, 45] = 10
    img[1, 1, 5, 40, 30] = 10

    # round 3
    img[2, 0, 5, 20, 40] = 10
    img[2, 0, 5, 40, 20] = 10

    # blur points
    gaussian_filter(img, (0, 0, 0.5, 1.5, 1.5), output=img)

    return ImageStack.from_numpy(img)


def multiple_possible_neighbors_with_jitter_with_noise() -> ImageStack:
    """last round has more spots and spots have some jitter <= 10px"""
    img = np.zeros((3, 2, 20, 50, 50), dtype=np.float32)

    # round 1
    img[0, 0, 5, 20, 40] = 10
    img[0, 1, 5, 40, 20] = 10

    # round 2
    img[1, 1, 5, 20, 45] = 10
    img[1, 1, 5, 30, 30] = 10

    # round 3
    img[2, 0, 5, 20, 40] = 10
    img[2, 0, 5, 30, 20] = 10
    img[2, 1, 5, 40, 30] = 1

    # blur points
    gaussian_filter(img, (0, 0, 0.5, 1.5, 1.5), output=img)

    return ImageStack.from_numpy(img)


def channels_crosstalk() -> ImageStack:
    """this code has spots with intensity crosstalk between channels of the same round"""
    img = np.zeros((3, 2, 20, 50, 50), dtype=np.float32)

    # round 1
    img[0, 0, 5, 20, 40] = 10
    img[0, 1, 5, 20, 40] = 5

    # round 2
    img[1, 0, 4, 20, 40] = 5
    img[1, 1, 5, 20, 40] = 10

    # round 3
    img[2, 0, 5, 20, 40] = 10

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


def simple_hmax_detector():
    return HMax(h=0.5)


h_max_detector = simple_hmax_detector()


def test_local_graph_blob_detector_empty_data():
    stack = empty_data()
    spots = h_max_detector.run(image_stack=stack)
    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=1,
        search_radius_max=1,
        anchor_round=0)

    assert intensity_table.shape == (0, 3, 2)
    f, c, r = np.nonzero(intensity_table.values)
    assert np.all(f == 0)
    assert np.all(c == 0)
    assert np.all(r == 0)


def test_local_graph_blob_detector_two_codes():
    stack = two_perfect_codes()
    # Find spots with 'h-maxima'
    spots = h_max_detector.run(image_stack=stack)
    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=1,
        search_radius_max=1,
        anchor_round=0)

    assert intensity_table.shape == (2, 2, 3)
    assert np.all(intensity_table[0][Axes.X.value] == 35)
    assert np.all(intensity_table[0][Axes.Y.value] == 20)
    assert np.all(intensity_table[0][Axes.ZPLANE.value] == 5)


def test_local_graph_blob_detector_jitter_code():
    stack = jitter_code()
    spots = h_max_detector.run(image_stack=stack)

    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=3,
        search_radius_max=3,
        anchor_round=0)

    assert intensity_table.shape == (1, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0, 0]))
    assert np.all(c == np.array([0, 0, 1]))
    assert np.all(r == np.array([0, 2, 1]))

    # test again with smaller search radius
    spots = h_max_detector.run(image_stack=stack)

    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=1,
        search_radius_max=5,
        anchor_round=0)

    assert intensity_table.shape == (0, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == 0)
    assert np.all(c == 0)
    assert np.all(r == 0)

    # test again with smaller search radius max
    spots = h_max_detector.run(image_stack=stack)

    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=3,
        search_radius_max=1,
        anchor_round=0)

    assert intensity_table.shape == (0, 2, 3)
    f, c, r = np.nonzero(intensity_table.values)
    assert np.all(f == 0)
    assert np.all(c == 0)
    assert np.all(r == 0)


def test_local_graph_blob_detector_traversing_code():
    stack = traversing_code()
    spots = h_max_detector.run(image_stack=stack)

    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=5,
        search_radius_max=10,
        anchor_round=0)

    assert intensity_table.shape == (1, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0, 0]))
    assert np.all(c == np.array([0, 0, 1]))
    assert np.all(r == np.array([0, 2, 1]))

    spots = h_max_detector.run(image_stack=stack)

    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=5,
        search_radius_max=5,
        anchor_round=0)

    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == 0)
    assert np.all(c == 0)
    assert np.all(r == 0)


def test_local_graph_blob_detector_multiple_neighbors():
    stack = multiple_possible_neighbors()
    spots = h_max_detector.run(image_stack=stack)
    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=4,
        search_radius_max=4,
        anchor_round=0)

    assert intensity_table.shape == (2, 2, 3)
    assert np.all(intensity_table[Axes.ZPLANE.value] == (5, 5))
    assert np.all(intensity_table[Axes.Y.value] == (20, 40))
    assert np.all(intensity_table[Axes.X.value] == (40, 20))

    spots = h_max_detector.run(image_stack=stack)
    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=15,
        search_radius_max=20,
        anchor_round=0)

    assert intensity_table.shape == (2, 2, 3)
    assert np.all(intensity_table[Axes.ZPLANE.value] == (5, 5))
    assert np.all(intensity_table[Axes.Y.value] == (20, 40))
    assert np.all(intensity_table[Axes.X.value] == (40, 20))


def test_local_graph_blob_detector_multiple_neighbors_with_jitter():
    stack = multiple_possible_neighbors_with_jitter()
    spots = h_max_detector.run(image_stack=stack)
    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=10,
        search_radius_max=20,
        anchor_round=0)

    assert intensity_table.shape == (2, 2, 3)
    assert np.all(intensity_table[Axes.ZPLANE.value] == (5, 5))
    assert np.all(intensity_table[Axes.Y.value] == (20, 40))
    assert np.all(intensity_table[Axes.X.value] == (40, 10))

    spots = h_max_detector.run(image_stack=stack)
    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=15,
        search_radius_max=15,
        anchor_round=0)

    assert intensity_table.shape == (1, 2, 3)
    assert np.all(intensity_table[Axes.ZPLANE.value] == (5))
    assert np.all(intensity_table[Axes.Y.value] == (20))
    assert np.all(intensity_table[Axes.X.value] == (40))


def test_local_graph_blob_detector_multiple_neighbors_with_jitter_with_noise():
    stack = multiple_possible_neighbors_with_jitter_with_noise()
    spots = h_max_detector.run(image_stack=stack)
    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=10,
        search_radius_max=20,
        anchor_round=0)

    assert intensity_table.shape == (2, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0, 0, 1, 1, 1]))
    assert np.all(c == np.array([0, 0, 1, 0, 1, 1]))
    assert np.all(r == np.array([0, 2, 1, 2, 0, 1]))


def test_local_graph_blob_detector_channels_crosstalk():
    stack = channels_crosstalk()
    spots = h_max_detector.run(image_stack=stack)
    intensity_table = build_traces_graph_based(
        spot_results=spots,
        k_d=0.33,
        search_radius=3,
        search_radius_max=5,
        anchor_round=0)

    assert intensity_table.shape == (1, 2, 3)
    f, c, r = np.where(~intensity_table.isnull())
    assert np.all(f == np.array([0, 0, 0, 0, 0]))
    assert np.all(c == np.array([0, 0, 0, 1, 1]))
    assert np.all(r == np.array([0, 1, 2, 0, 1]))
