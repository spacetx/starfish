import numpy as np
import pytest
from skimage.feature import blob_log

from starfish.spots._detector._base import SpotFinderAlgorithmBase
from starfish.spots._detector.detect import detect_spots
from starfish.spots._detector.gaussian import GaussianSpotDetector
from starfish.spots._detector.local_max_peak_finder import LocalMaxPeakFinder
from starfish.stack import ImageStack
# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import (
    synthetic_dataset_with_truth_values,
    synthetic_dataset_with_truth_values_and_called_spots,
    synthetic_single_spot_2d,
    synthetic_single_spot_3d,
    synthetic_single_spot_imagestack_2d,
    synthetic_single_spot_imagestack_3d,
    synthetic_two_spot_3d,
    synthetic_two_spot_3d_2round_2ch,
    synthetic_two_spot_imagestack_3d,
)
from starfish.types import Indices


def test_blob_log_2d(synthetic_single_spot_2d):
    """Tests that skimage.feature.blob_log is acting as we expect on 2d data"""
    result = blob_log(synthetic_single_spot_2d, min_sigma=2, max_sigma=4, num_sigma=10, threshold=0)
    assert np.array_equal(result, np.array([[10, 90, 2]]))


def test_blob_log_3d(synthetic_single_spot_3d):
    """Tests that skimage.feature.blob_log is acting as we expect on 3d data"""
    result = blob_log(synthetic_single_spot_3d, min_sigma=2, max_sigma=4, num_sigma=10, threshold=0)
    assert np.array_equal(result, np.array([[5, 10, 90, 2]]))


def test_blob_log_3d_flat(synthetic_single_spot_2d):
    """
    Tests that skimage.feature.blob_log is acting as we expect on pseudo-3d data where the
    z-axis has size 1
    """
    data = synthetic_single_spot_2d.reshape(1, *synthetic_single_spot_2d.shape)
    result = blob_log(data, min_sigma=2, max_sigma=4, num_sigma=10, threshold=0)
    assert np.array_equal(result, np.array([[0, 10, 90, 2]]))


def simple_gaussian_spot_detector() -> GaussianSpotDetector:
    """create a basic gaussian spot detector"""
    return GaussianSpotDetector(
        min_sigma=1,
        max_sigma=4,
        num_sigma=5,
        threshold=0,
        measurement_type='max',
    )


def simple_local_max_spot_detector() -> LocalMaxPeakFinder:
    """create a basic local max peak finder"""
    return LocalMaxPeakFinder(
        spot_diameter=3,
        min_mass=5,
        max_size=10,
        separation=2,
    )


# initialize spot detectors
gaussian_spot_detector = simple_gaussian_spot_detector()
local_max_spot_detector = simple_local_max_spot_detector()

# create the data_stack
data_stack = synthetic_two_spot_3d_2round_2ch()


@pytest.mark.parametrize('data_stack, spot_detector, radius_is_gyration', [
    (data_stack, gaussian_spot_detector, False),
    (data_stack, local_max_spot_detector, True)
])
def test_spot_detection_with_reference_image(
        data_stack: ImageStack,
        spot_detector: SpotFinderAlgorithmBase,
        radius_is_gyration: bool,
):
    """
    This testing method uses a reference image to identify spot locations. Thus, it should detect
    two spots, each with max intensity 7. Because the channels and rounds are aggregated, this
    method should recognize the 1-hot code used in the testing data, and see one channel "on" per
    round. Thus, the total intensity across all channels and round for each spot should be 14.

    """
    reference_image = data_stack.max_proj(
        Indices.CH, Indices.ROUND)

    intensity_table = detect_spots(
        data_stack=data_stack,
        spot_finding_method=spot_detector.image_to_spots,
        reference_image=reference_image,
        measurement_function=np.max,
        radius_is_gyration=radius_is_gyration,
    )
    assert intensity_table.shape == (2, 2, 2), "wrong number of spots detected"
    expected = [14, 14]
    assert np.array_equal(intensity_table.sum((Indices.ROUND, Indices.CH)), expected), \
        "wrong spot intensities detected"


@pytest.mark.parametrize('data_stack, spot_detector, radius_is_gyration', [
    (data_stack, gaussian_spot_detector, False),
    (data_stack, local_max_spot_detector, True)
])
def test_spot_detection_with_reference_image_from_max_projection(
        data_stack: ImageStack,
        spot_detector: SpotFinderAlgorithmBase,
        radius_is_gyration: bool,
):
    """
    This testing method builds a reference image to identify spot locations. Thus, it should detect
    two spots, each with max intensity 7. Because the channels and rounds are aggregated, this
    method should recognize the 1-hot code used in the testing data, and see one channel "on" per
    round. Thus, the total intensity across all channels and round for each spot should be 14.
    """
    intensity_table = detect_spots(
        data_stack=data_stack,
        spot_finding_method=spot_detector.image_to_spots,
        reference_image_from_max_projection=True,
        measurement_function=np.max,
        radius_is_gyration=radius_is_gyration,
    )
    assert intensity_table.shape == (2, 2, 2), "wrong number of spots detected"
    expected = [14, 14]
    assert np.array_equal(intensity_table.sum((Indices.ROUND, Indices.CH)), expected), \
        "wrong spot intensities detected"


@pytest.mark.parametrize('data_stack, spot_detector, radius_is_gyration', [
    (data_stack, gaussian_spot_detector, False),
    (data_stack, local_max_spot_detector, True)
])
def test_spot_finding_no_reference_image(
        data_stack: ImageStack,
        spot_detector: SpotFinderAlgorithmBase,
        radius_is_gyration: bool,
):
    """
    This testing method does not provide a reference image, and should therefore check for spots
    in each (round, ch) combination in sequence. With the given input, it should detect 4 spots,
    each with a max value of 7. Because each (round, ch) are measured sequentially, each spot only
    measures a single channel. Thus the total intensity across all rounds and channels for each
    spot should be 7.
    """
    intensity_table = detect_spots(
        data_stack=data_stack,
        spot_finding_method=spot_detector.image_to_spots,
        measurement_function=np.max,
        radius_is_gyration=radius_is_gyration,
    )
    assert intensity_table.shape == (4, 2, 2), "wrong number of spots detected"
    expected = [7, 7, 7, 7]
    assert np.array_equal(intensity_table.sum((Indices.ROUND, Indices.CH)), expected), \
        "wrong spot intensities detected"
