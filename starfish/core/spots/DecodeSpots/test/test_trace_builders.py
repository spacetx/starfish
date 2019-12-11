import numpy as np
import pytest

from starfish import ImageStack
from starfish.core.spots.DecodeSpots import trace_builders
from starfish.core.spots.FindSpots import BlobDetector
from starfish.core.spots.FindSpots import FindSpotsAlgorithm
from starfish.core.test.factories import (
    two_spot_informative_blank_coded_data_factory,
    two_spot_one_hot_coded_data_factory,
    two_spot_sparse_coded_data_factory,
)
from starfish.types import Axes, Features, FunctionSource


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


# initialize spot detectors
gaussian_spot_detector = simple_gaussian_spot_detector()


# test parameterization
test_parameters = (
    'data_stack, spot_detector, max_intensity',
    [
        (ONE_HOT_IMAGESTACK, gaussian_spot_detector, ONE_HOT_MAX_INTENSITY),
        (SPARSE_IMAGESTACK, gaussian_spot_detector, SPARSE_MAX_INTENSITY),
        (BLANK_IMAGESTACK, gaussian_spot_detector, BLANK_MAX_INTENSITY),
    ]
)


@pytest.mark.parametrize(*test_parameters)
def test_spot_detection_with_reference_image_exact_match(
        data_stack: ImageStack,
        spot_detector: FindSpotsAlgorithm,
        max_intensity: float,
):
    """This testing method uses a reference image to identify spot locations then builds traces
    using the exact_match strategy. This represents a workflow common in a multiplexed assays.
    Each method should detect 2 total spots in the max projected reference image then group them
    into 2 distinct spot traces across the ImageStack. """

    reference_image = data_stack.reduce((Axes.ROUND, Axes.CH), func=FunctionSource.np("max"))
    spots = spot_detector.run(image_stack=data_stack, reference_image=reference_image)
    intensity_table = trace_builders.build_spot_traces_exact_match(spots)
    assert intensity_table.sizes[Features.AXIS] == 2, "wrong number of spots traces detected"
    expected = [max_intensity * 2, max_intensity * 2]
    assert np.allclose(intensity_table.sum((Axes.ROUND, Axes.CH)).values, expected), \
        "wrong spot intensities detected"

    # verify this execution strategy produces an empty intensitytable when called with a blank image
    reference_image = EMPTY_IMAGESTACK.reduce((Axes.ROUND, Axes.CH), func=FunctionSource.np("max"))
    spots = spot_detector.run(image_stack=EMPTY_IMAGESTACK, reference_image=reference_image)
    empty_intensity_table = trace_builders.build_spot_traces_exact_match(spots)
    assert empty_intensity_table.sizes[Features.AXIS] == 0


@pytest.mark.parametrize(*test_parameters)
def test_spot_detection_no_reference_image_exact_match(
        data_stack: ImageStack,
        spot_detector: FindSpotsAlgorithm,
        max_intensity: float,
):
    """
    This testing method does not provide a reference image, and should therefore check for spots
    in each (round, ch) combination in sequence. It then uses the exact_match strategy to build
    spot traces. Since the spot finding only finds 1 spot in the first tile, the trace builder
    only builds one trace. This workflow doesn't really make real world sense but we're testing it
    anyway.
    """
    spots = spot_detector.run(image_stack=data_stack)
    intensity_table = trace_builders.build_spot_traces_exact_match(spots)
    assert intensity_table.sizes[Features.AXIS] == 1, "wrong number of spots traces detected"

    spots = spot_detector.run(image_stack=EMPTY_IMAGESTACK)
    empty_intensity_table = trace_builders.build_spot_traces_exact_match(spots)
    assert empty_intensity_table.sizes[Features.AXIS] == 0


@pytest.mark.parametrize(*test_parameters)
def test_spot_finding_no_reference_image_sequential(
        data_stack: ImageStack,
        spot_detector: FindSpotsAlgorithm,
        max_intensity: float,
):
    """
    This testing method does not provide a reference image, and should therefore check for spots
    in each (round, ch) combination in sequence. With the given input, it should detect 4 spots.
    It then uses the sequential trace building strategy which should treat each spot as it's own
    unique trace totalling 4 traces.
    """
    spots = spot_detector.run(image_stack=data_stack)
    intensity_table = trace_builders.build_traces_sequential(spots)
    assert intensity_table.sizes[Features.AXIS] == 4, "wrong number of spot traces detected"
    expected = [max_intensity] * 4
    assert np.allclose(intensity_table.sum((Axes.ROUND, Axes.CH)).values, expected), \
        "wrong spot intensities detected"

    spots = spot_detector.run(image_stack=EMPTY_IMAGESTACK)
    empty_intensity_table = trace_builders.build_traces_sequential(spots)
    assert empty_intensity_table.sizes[Features.AXIS] == 0


@pytest.mark.parametrize(*test_parameters)
def test_spot_finding_reference_image_sequential(
        data_stack: ImageStack,
        spot_detector: FindSpotsAlgorithm,
        max_intensity: float,
):
    """
    This testing method uses a reference image to identify spot locations then builds traces
    using the sequential strategy. It finds 2 spots in the max projected image, then measures the
    two spots on each tile totally 2*num_chs*num_rounds spots. When building spot traces it treats
    each spot as it's own trace totally 8 traces. This workflow doesn't really make sense but
    we're testing it anyway.
    """

    reference_image = data_stack.reduce((Axes.ROUND, Axes.CH), func=FunctionSource.np("max"))
    spots = spot_detector.run(image_stack=data_stack, reference_image=reference_image)
    intensity_table = trace_builders.build_traces_sequential(spots)
    expected_num_traces = (2 * data_stack.num_chs * data_stack.num_rounds)
    assert intensity_table.sizes[Features.AXIS] == expected_num_traces, "wrong number of " \
                                                                        "spots traces detected"

    reference_image = EMPTY_IMAGESTACK.reduce((Axes.ROUND, Axes.CH), func=FunctionSource.np("max"))
    spots = spot_detector.run(image_stack=EMPTY_IMAGESTACK, reference_image=reference_image)
    empty_intensity_table = trace_builders.build_traces_sequential(spots)
    assert empty_intensity_table.sizes[Features.AXIS] == 0
