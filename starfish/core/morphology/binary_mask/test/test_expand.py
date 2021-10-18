import numpy as np
import pytest

from ..expand import fill_from_mask


def test_expand_mask_2d(shape=(15, 15)):
    mask = np.array(
        [[True, False, False],
         [False, True, True]],
        dtype=bool)
    label_image = np.random.randint(30, np.iinfo(np.uint16).max, size=shape, dtype=np.uint16)
    label_image_clone = np.copy(label_image)
    fill_from_mask(mask, (2, 3), 20, label_image)

    assert label_image.dtype == np.uint16
    assert label_image.shape == shape

    compare_mask = np.ones(shape=shape, dtype=bool)
    for y, x in (
            (2, 3),
            (3, 4),
            (3, 5),
    ):
        assert label_image[y, x] == 20
        compare_mask[y, x] = False

    assert np.all(label_image_clone & compare_mask == label_image & compare_mask)


def test_expand_mask_3d(shape=(15, 15, 15)):
    mask = np.array(
        [[[True, False, True],
          [True, False, True],
          [False, False, False],
          [False, True, False]],

         [[True, False, True],
          [False, False, False],
          [True, False, False],
          [False, False, False]]],
        dtype=bool)
    label_image = np.random.randint(30, np.iinfo(np.uint16).max, size=shape, dtype=np.uint16)
    label_image_clone = np.copy(label_image)
    fill_from_mask(mask, (2, 7, 3), 20, label_image)

    assert label_image.dtype == np.uint16
    assert label_image.shape == shape

    compare_mask = np.ones(shape=shape, dtype=bool)
    for z, y, x in (
            (2, 7, 3),
            (2, 7, 5),
            (2, 8, 3),
            (2, 8, 5),
            (2, 10, 4),
            (3, 7, 3),
            (3, 7, 5),
            (3, 9, 3),
    ):
        assert label_image[z, y, x] == 20
        compare_mask[z, y, x] = False

    assert np.all(label_image_clone & compare_mask == label_image & compare_mask)


def test_expand_image_too_small(shape=(2, 2)):
    mask = np.array(
        [[True, False, False],
         [False, True, True]],
        dtype=bool)
    label_image = np.random.randint(30, np.iinfo(np.uint16).max, size=shape, dtype=np.uint16)
    with pytest.raises(ValueError):
        fill_from_mask(mask, (2, 3), 20, label_image)


def test_expand_image_negative_axis_labels(shape=(15, 15)):
    mask = np.array(
        [[True, False, False],
         [False, True, True]],
        dtype=bool)
    label_image = np.random.randint(30, np.iinfo(np.uint16).max, size=shape, dtype=np.uint16)
    with pytest.raises(ValueError):
        fill_from_mask(mask, (2, -1), 20, label_image)
