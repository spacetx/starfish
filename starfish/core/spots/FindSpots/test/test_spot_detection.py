import numpy as np
import pytest

from starfish import ImageStack
from starfish.core.imagestack.test.factories import unique_tiles_imagestack
from starfish.core.test.factories import (
    two_spot_informative_blank_coded_data_factory,
    two_spot_one_hot_coded_data_factory,
    two_spot_sparse_coded_data_factory,
)
from starfish.types import Axes, FunctionSource
from .._base import FindSpotsAlgorithm
from ..blob import BlobDetector
from ..local_max_peak_finder import LocalMaxPeakFinder
from ..trackpy_local_max_peak_finder import TrackpyLocalMaxPeakFinder

# verify all spot finders handle different coding types
_, ONE_HOT_IMAGESTACK, ONE_HOT_MAX_INTENSITY = two_spot_one_hot_coded_data_factory()
_, SPARSE_IMAGESTACK, SPARSE_MAX_INTENSITY = two_spot_sparse_coded_data_factory()
_, BLANK_IMAGESTACK, BLANK_MAX_INTENSITY = two_spot_informative_blank_coded_data_factory()

# make sure that all spot finders handle empty arrays
EMPTY_IMAGESTACK = ImageStack.from_numpy(np.zeros((4, 2, 10, 100, 100), dtype=np.float32))


def simple_gaussian_spot_detector() -> BlobDetector:
    """create a basic gaussian spot detector"""
    return BlobDetector(
        min_sigma=1,
        max_sigma=4,
        num_sigma=5,
        threshold=0,
        measurement_type='max')


def simple_trackpy_local_max_spot_detector() -> TrackpyLocalMaxPeakFinder:
    """create a basic local max peak finder"""
    return TrackpyLocalMaxPeakFinder(
        spot_diameter=3,
        min_mass=0.01,
        max_size=10,
        separation=2,
        radius_is_gyration=True
    )


def simple_local_max_spot_detector_3d() -> LocalMaxPeakFinder:
    return LocalMaxPeakFinder(
        min_distance=6,
        stringency=0,
        min_obj_area=0,
        max_obj_area=np.iinfo(int).max,
        threshold=0,
        is_volume=True,
    )


def simple_local_max_spot_detector_2d() -> LocalMaxPeakFinder:
    return LocalMaxPeakFinder(
        min_distance=6,
        stringency=0,
        min_obj_area=0,
        max_obj_area=np.iinfo(int).max,
        threshold=0,
        is_volume=False
    )


# initialize spot detectors
gaussian_spot_detector = simple_gaussian_spot_detector()
trackpy_local_max_spot_detector = simple_trackpy_local_max_spot_detector()
local_max_spot_detector_2d = simple_local_max_spot_detector_2d()
local_max_spot_detector_3d = simple_local_max_spot_detector_3d()

# test parameterization
test_parameters = (
    'data_stack, spot_detector, is_2d',
    [
        (ONE_HOT_IMAGESTACK, gaussian_spot_detector, False),
        (ONE_HOT_IMAGESTACK, trackpy_local_max_spot_detector, False),
        (ONE_HOT_IMAGESTACK, local_max_spot_detector_2d, True),
        (ONE_HOT_IMAGESTACK, local_max_spot_detector_3d, False),
        (SPARSE_IMAGESTACK, gaussian_spot_detector, False),
        (SPARSE_IMAGESTACK, trackpy_local_max_spot_detector, False),
        (SPARSE_IMAGESTACK, local_max_spot_detector_2d, True),
        (SPARSE_IMAGESTACK, local_max_spot_detector_3d, False),
        (BLANK_IMAGESTACK, gaussian_spot_detector, False),
        (BLANK_IMAGESTACK, trackpy_local_max_spot_detector, False),
        (BLANK_IMAGESTACK, local_max_spot_detector_2d, True),
        (BLANK_IMAGESTACK, local_max_spot_detector_3d, False),
    ]
)


@pytest.mark.parametrize(*test_parameters)
def test_spot_detection_with_reference_image(
        data_stack: ImageStack,
        spot_detector: FindSpotsAlgorithm,
        is_2d: bool,
):
    """This testing method uses a reference image to identify spot locations. Each method should
    find 2 spots in the reference image then measure the intensity of those locations in each
    r/ch pair. The final spot results should represent 2 spots for each r/ch totalling
    2*num_rounds*num_ch spots """

    # 2D spot detection will find spots on 8 zlayers, so it produces 7x as many spots.
    multiplier = 7 if is_2d else 1
    reference_image = data_stack.reduce((Axes.ROUND, Axes.CH), func=FunctionSource.np("max"))
    spots_results = spot_detector.run(image_stack=data_stack, reference_image=reference_image)
    assert spots_results.count_total_spots() == (
        2 * data_stack.num_chs * data_stack.num_rounds * multiplier), \
        "wrong number of spots detected"


@pytest.mark.parametrize(*test_parameters)
def test_spot_detection_no_reference_image(
        data_stack: ImageStack,
        spot_detector: FindSpotsAlgorithm,
        is_2d: bool,
):
    """
    This testing method does not provide a reference image, and should therefore check for spots
    in each (round, ch) combination in sequence. The final spot results should represent the
    total number of original spots in the ImageStack.
    """

    # 2D spot detection will find spots on 8 zlayers, so it produces 7x as many spots.
    multiplier = 7 if is_2d else 1
    spots_results = spot_detector.run(image_stack=data_stack)
    assert spots_results.count_total_spots() == 4 * multiplier, "wrong number of spots detected"

    spots = spot_detector.run(image_stack=EMPTY_IMAGESTACK)
    assert spots.count_total_spots() == 0


def _make_labeled_image() -> ImageStack:
    ROUND_LABELS = (1, 4, 6)
    CH_LABELS = (2, 4, 6, 8)
    ZPLANE_LABELS = (3, 4)
    HEIGHT = 2
    WIDTH = 4

    return unique_tiles_imagestack(
        ROUND_LABELS, CH_LABELS, ZPLANE_LABELS, HEIGHT, WIDTH)


def test_reference_image_spot_detection_with_image_with_labeled_axes():
    """This testing method uses a reference image to identify spot locations."""
    data_stack = _make_labeled_image()
    reference_image = data_stack.reduce((Axes.ROUND, Axes.CH), func=FunctionSource.np("max"))
    spot_results = gaussian_spot_detector.run(image_stack=data_stack,
                                              reference_image=reference_image)
    return spot_results


def test_spot_detection_with_image_with_labeled_axes():
    """This testing method uses no reference image to identify spot locations."""
    data_stack = _make_labeled_image()
    spot_results = gaussian_spot_detector.run(image_stack=data_stack)
    return spot_results
