import numpy as np
import pytest
from skimage.feature import blob_log

from starfish.constants import Features
from starfish.constants import Indices
from starfish.pipeline.features.spots.detector.detect import (
    detect_spots,
    measure_spot_intensities,
)
from starfish.pipeline.features.spots.detector.gaussian import (
    GaussianSpotDetector,
    gaussian_spot_detector
)
# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import (
    synthetic_dataset_with_truth_values_and_called_spots, synthetic_single_spot_2d,
    synthetic_two_spot_3d, synthetic_single_spot_imagestack_2d, synthetic_single_spot_imagestack_3d,
    synthetic_two_spot_imagestack_3d, synthetic_dataset_with_truth_values, synthetic_single_spot_3d,
    synthetic_two_spot_3d_2round_2ch
)


def test_spots_match_coordinates_of_synthesized_spots(
        synthetic_dataset_with_truth_values_and_called_spots):

    codebook, true_intensities, image, intensities = (
        synthetic_dataset_with_truth_values_and_called_spots)
    assert true_intensities.shape == intensities.shape

    # we do a bit of rounding, some of these may be off with 1-pixel error.
    x_matches = np.all(np.abs(np.sort(true_intensities.coords[Features.X].values) -
                              np.sort(intensities.coords[Features.X].values)) <= 1)
    y_matches = np.all(np.abs(np.sort(true_intensities.coords[Features.Y].values) -
                              np.sort(intensities.coords[Features.Y].values)) <= 1)
    assert x_matches
    assert y_matches


def test_create_intensity_table(synthetic_dataset_with_truth_values_and_called_spots):
    codebook, true_intensities, image, intensities = (
        synthetic_dataset_with_truth_values_and_called_spots)

    assert intensities.shape[0] == 5


def test_create_intensity_table_raises_value_error_when_no_spots_detected(
        synthetic_dataset_with_truth_values_and_called_spots):

    codebook, true_intensities, image, intensities = (
        synthetic_dataset_with_truth_values_and_called_spots)

    min_sigma = 1
    max_sigma = 10
    num_sigma = 30
    threshold = 1000000  # no blobs will be above this high threshold

    gsd = GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        blobs_stack=image,
        measurement_type='max',
    )
    with pytest.raises(ValueError):
        gsd.find(image_stack=image)


def test_blob_log_2d(synthetic_single_spot_2d):
    result = blob_log(synthetic_single_spot_2d, min_sigma=2, max_sigma=4, num_sigma=10, threshold=0)
    assert np.array_equal(result, np.array([[10, 90, 2]]))


def test_blob_log_3d(synthetic_single_spot_3d):
    result = blob_log(synthetic_single_spot_3d, min_sigma=2, max_sigma=4, num_sigma=10, threshold=0)
    assert np.array_equal(result, np.array([[5, 10, 90, 2]]))


def test_blob_log_3d_flat(synthetic_single_spot_2d):
    """
    verify that 3d blob log works, even when the third dimension is too small to support the
    observed standard deviation
    """
    data = synthetic_single_spot_2d.reshape(1, *synthetic_single_spot_2d.shape)
    result = blob_log(data, min_sigma=2, max_sigma=4, num_sigma=10, threshold=0)
    assert np.array_equal(result, np.array([[0, 10, 90, 2]]))


def single_spot_detector(stack):
    gsd = GaussianSpotDetector(
        min_sigma=1,
        max_sigma=4,
        num_sigma=5,
        threshold=0,
        blobs_stack=stack,
        measurement_type='max',
    )
    return gsd


def test_find_two_spots_3d(synthetic_two_spot_imagestack_3d):
    gsd = single_spot_detector(synthetic_two_spot_imagestack_3d)
    intensity_table = gsd.find(gsd.blobs_stack)

    assert intensity_table.shape == (2, 1, 1)
    expected = np.max(gsd.blobs_image).repeat(2).reshape(2, 1, 1)
    assert np.array_equal(intensity_table.values, expected)


def test_refactored_gsd(synthetic_two_spot_3d_2round_2ch):
    """
    This testing method does not provide a reference image, and should therefore check for spots
    in each (round, ch) combination in sequence. With the given input, it should detect 4 spots,
    each with a max value of 7. Because each (round, ch) are measured sequentially, each spot only
    measures a single channel. Thus the total intensity across all rounds and channels for each
    spot should be 7.
    """
    spot_finding_kwargs = dict(
        min_sigma=1,
        max_sigma=4,
        num_sigma=5,
        threshold=0,
        measurement_function=np.max
    )
    intensity_table = detect_spots(
        data_image=synthetic_two_spot_3d_2round_2ch,
        spot_finding_method=gaussian_spot_detector,
        spot_finding_kwargs=spot_finding_kwargs,
        measurement_function=np.max
    )
    assert intensity_table.shape == (4, 2, 2)
    expected = [7, 7, 7, 7]
    assert np.array_equal(intensity_table.sum((Indices.ROUND, Indices.CH)), expected)


def test_refactored_gsd_with_reference_image(synthetic_two_spot_3d_2round_2ch):
    """
    This testing method uses a reference image to identify spot locations. Thus, it should detect
    two spots, each with max intensity 7. Because the channels and rounds are aggregated, this
    method should recognize the 1-hot code used in the testing data, and see one channel "on" per
    round. Thus, the total intensity across all channels and round for each spot should be 14.

    """
    reference_image = synthetic_two_spot_3d_2round_2ch.max_proj(
        Indices.CH, Indices.ROUND)
    spot_finding_kwargs = dict(
        min_sigma=1,
        max_sigma=4,
        num_sigma=5,
        threshold=0,
        measurement_function=np.max
    )
    intensity_table = detect_spots(
        data_image=synthetic_two_spot_3d_2round_2ch,
        spot_finding_method=gaussian_spot_detector,
        spot_finding_kwargs=spot_finding_kwargs,
        reference_image=reference_image,
        measurement_function=np.max
    )
    assert intensity_table.shape == (2, 2, 2)
    expected = [14, 14]
    assert np.array_equal(intensity_table.sum((Indices.ROUND, Indices.CH)), expected)


def test_refactored_gsd_with_reference_image_from_max_projection(synthetic_two_spot_3d_2round_2ch):
    """
    This testing method builds a reference image to identify spot locations. Thus, it should detect
    two spots, each with max intensity 7. Because the channels and rounds are aggregated, this
    method should recognize the 1-hot code used in the testing data, and see one channel "on" per
    round. Thus, the total intensity across all channels and round for each spot should be 14.
    """

    spot_finding_kwargs = dict(
        min_sigma=1,
        max_sigma=4,
        num_sigma=5,
        threshold=0,
        measurement_function=np.max
    )
    intensity_table = detect_spots(
        data_image=synthetic_two_spot_3d_2round_2ch,
        spot_finding_method=gaussian_spot_detector,
        spot_finding_kwargs=spot_finding_kwargs,
        reference_from_max_projection=True,
        measurement_function=np.max
    )
    assert intensity_table.shape == (2, 2, 2)
    expected = [14, 14]
    assert np.array_equal(intensity_table.sum((Indices.ROUND, Indices.CH)), expected)
