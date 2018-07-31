import numpy as np
import pytest
from skimage.feature import blob_log

from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.constants import Features
# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import (
    synthetic_dataset_with_truth_values_and_called_spots, synthetic_single_spot_2d,
    synthetic_two_spot_3d, synthetic_single_spot_imagestack_2d, synthetic_single_spot_imagestack_3d,
    synthetic_two_spot_imagestack_3d, synthetic_dataset_with_truth_values, synthetic_single_spot_3d)


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


def test_find_spot_locations_2d(synthetic_single_spot_imagestack_2d):

    gsd = single_spot_detector(synthetic_single_spot_imagestack_2d)
    # noinspection PyProtectedMember
    spot_attributes = gsd._find_spot_locations()

    assert spot_attributes.shape[0] == 1
    assert np.array_equal(spot_attributes[[Features.Z, Features.Y, Features.X]].values,
                          np.array([[0, 10, 90]]))
    # rounding incurs an error of up to one pixel
    tol = 1
    assert np.all(
        np.abs(spot_attributes[['z_min', 'z_max', 'y_min', 'y_max', 'x_min', 'x_max']].values -
               np.array([[0, 1, 7, 13, 87, 93]])) <= tol
    )


# noinspection PyProtectedMember
def test_find_spot_locations_3d(synthetic_single_spot_imagestack_3d):
    gsd = single_spot_detector(synthetic_single_spot_imagestack_3d)
    spot_attributes = gsd._find_spot_locations()

    assert spot_attributes.shape[0] == 1
    assert np.array_equal(spot_attributes[[Features.Z, Features.Y, Features.X]].values,
                          np.array([[5, 10, 90]]))
    # rounding incurs an error of up to one pixel
    tol = 1
    assert np.all(
        np.abs(spot_attributes[['z_min', 'z_max', 'y_min', 'y_max', 'x_min', 'x_max']].values -
               np.array([[2, 8, 7, 13, 87, 93]])) <= tol
    )


# noinspection PyProtectedMember
def test_measure_spot_intensity_2d(synthetic_single_spot_imagestack_2d):
    gsd = single_spot_detector(synthetic_single_spot_imagestack_2d)
    spot_attributes = gsd._find_spot_locations()
    intensity_table = gsd._measure_spot_intensities(
        synthetic_single_spot_imagestack_2d, spot_attributes)

    assert intensity_table.shape == (1, 1, 1)
    assert np.array_equal(intensity_table.values, np.array([[[np.max(gsd.blobs_image)]]]))


# noinspection PyProtectedMember
def test_measure_spot_intensity_3d(synthetic_single_spot_imagestack_3d):
    gsd = single_spot_detector(synthetic_single_spot_imagestack_3d)
    spot_attributes = gsd._find_spot_locations()
    intensity_table = gsd._measure_spot_intensities(
        synthetic_single_spot_imagestack_3d, spot_attributes)

    assert intensity_table.shape == (1, 1, 1)
    assert np.array_equal(intensity_table.values, np.array([[[np.max(gsd.blobs_image)]]]))


# noinspection PyProtectedMember
def test_measure_two_spots_3d(synthetic_two_spot_imagestack_3d):
    gsd = single_spot_detector(synthetic_two_spot_imagestack_3d)
    spot_attributes = gsd._find_spot_locations()
    intensity_table = gsd._measure_spot_intensities(
        synthetic_two_spot_imagestack_3d, spot_attributes)

    assert intensity_table.shape == (2, 1, 1)
    expected = np.max(gsd.blobs_image).repeat(2).reshape(2, 1, 1)
    assert np.array_equal(intensity_table.values, expected)


def test_find_two_spots_3d(synthetic_two_spot_imagestack_3d):
    gsd = single_spot_detector(synthetic_two_spot_imagestack_3d)
    intensity_table = gsd.find(gsd.blobs_stack)

    assert intensity_table.shape == (2, 1, 1)
    expected = np.max(gsd.blobs_image).repeat(2).reshape(2, 1, 1)
    assert np.array_equal(intensity_table.values, expected)
