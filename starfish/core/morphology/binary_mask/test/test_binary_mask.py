import numpy as np
from skimage.morphology import binary_dilation

from starfish.core.morphology.label_image import LabelImage
from starfish.core.types import Axes, Coordinates
from .factories import binary_mask_collection_2d, label_array_2d
from ..binary_mask import BinaryMaskCollection


def test_from_label_image():
    label_image_array, physical_ticks = label_array_2d()

    label_image = LabelImage.from_label_array_and_ticks(
        label_image_array,
        None,
        physical_ticks,
        None,
    )

    mask_collection = BinaryMaskCollection.from_label_image(label_image)
    assert len(mask_collection) == 2

    region_0, region_1 = mask_collection.masks()

    assert region_0.name == '0'
    assert region_1.name == '1'

    assert np.array_equal(region_0, np.ones((1, 6), dtype=np.bool))
    temp = np.ones((2, 3), dtype=np.bool)
    temp[-1, -1] = False
    assert np.array_equal(region_1, temp)

    assert np.array_equal(region_0[Axes.Y.value], [0])
    assert np.array_equal(region_0[Axes.X.value], [0, 1, 2, 3, 4, 5])

    assert np.array_equal(region_1[Axes.Y.value], [3, 4])
    assert np.array_equal(region_1[Axes.X.value], [3, 4, 5])

    assert np.array_equal(region_0[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][0:1])
    assert np.array_equal(region_0[Coordinates.X.value],
                          physical_ticks[Coordinates.X][0:6])

    assert np.array_equal(region_1[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][3:5])
    assert np.array_equal(region_1[Coordinates.X.value],
                          physical_ticks[Coordinates.X][3:6])


def test_uncropped_mask():
    """Test that BinaryMaskCollection.uncropped_mask() works correctly.
    """
    label_image_array, physical_ticks = label_array_2d()

    label_image = LabelImage.from_label_array_and_ticks(
        label_image_array,
        None,
        physical_ticks,
        None,
    )

    mask_collection = BinaryMaskCollection.from_label_image(label_image)
    assert len(mask_collection) == 2

    region_0 = mask_collection.uncropped_mask(0)
    assert region_0.shape == label_image_array.shape
    assert region_0.dtype == np.bool
    assert np.all(region_0[0] == 1)
    assert np.all(region_0[1:5] == 0)

    region_1 = mask_collection.uncropped_mask(1)
    assert region_1.shape == label_image_array.shape
    assert region_1.dtype == np.bool
    assert np.all(region_1[0:3, :] == 0)
    assert np.all(region_1[:, 0:3] == 0)
    assert np.all(region_1[3:5, 3:6] == [[1, 1, 1],
                                         [1, 1, 0]])


def test_uncropped_mask_no_uncropping():
    """If the mask doesn't need to be uncropped, it should still work.  This is an optimized code
    path, so it is separately validated.
    """
    label_image_array, physical_ticks = label_array_2d()
    label_image_array.fill(1)

    label_image = LabelImage.from_label_array_and_ticks(
        label_image_array,
        None,
        physical_ticks,
        None,
    )

    mask_collection = BinaryMaskCollection.from_label_image(label_image)
    assert len(mask_collection) == 1

    region = mask_collection.uncropped_mask(0)
    assert region.shape == label_image_array.shape
    assert np.all(region == 1)


def test_to_label_image():
    # test via roundtrip
    label_image_array, physical_ticks = label_array_2d()

    label_image = LabelImage.from_label_array_and_ticks(
        label_image_array,
        None,
        physical_ticks,
        None,
    )

    masks = BinaryMaskCollection.from_label_image(label_image)

    assert np.array_equal(masks.to_label_image().xarray, label_image.xarray)


def test_save_load(tmp_path):
    binary_mask_collection = binary_mask_collection_2d()

    path = tmp_path / "data.tgz"
    binary_mask_collection.to_targz(path)
    masks2 = BinaryMaskCollection.open_targz(path)
    for m, m2 in zip(binary_mask_collection.masks(), masks2.masks()):
        assert np.array_equal(m, m2)

    # ensure that the regionprops are equal
    for ix in range(len(binary_mask_collection)):
        original_props = binary_mask_collection.mask_regionprops(ix)
        recalculated_props = binary_mask_collection.mask_regionprops(ix)
        assert original_props == recalculated_props


def test_from_empty_label_image(tmp_path):
    label_array, physical_ticks = label_array_2d()
    label_array.fill(0)

    label_image = LabelImage.from_label_array_and_ticks(
        label_array,
        None,
        physical_ticks,
        None,
    )

    binary_mask_collection = BinaryMaskCollection.from_label_image(label_image)
    masks = list(binary_mask_collection.masks())

    assert len(masks) == 0

    path = tmp_path / "data.tgz"
    binary_mask_collection.to_targz(path)
    masks2 = BinaryMaskCollection.open_targz(path)
    for m, m2 in zip(binary_mask_collection.masks(), masks2.masks()):
        assert np.array_equal(m, m2)

    # ensure that the regionprops are equal
    for ix in range(len(binary_mask_collection)):
        original_props = binary_mask_collection.mask_regionprops(ix)
        recalculated_props = binary_mask_collection.mask_regionprops(ix)
        assert original_props == recalculated_props


def test_apply():
    input_mask_collection = binary_mask_collection_2d()
    output_mask_collection = input_mask_collection._apply(binary_dilation)

    assert input_mask_collection._pixel_ticks == output_mask_collection._pixel_ticks
    assert input_mask_collection._physical_ticks == output_mask_collection._physical_ticks
    assert input_mask_collection._log == output_mask_collection._log
    assert len(input_mask_collection) == len(output_mask_collection)

    region_0, region_1 = output_mask_collection.masks()

    assert region_0.name == '0'
    assert region_1.name == '1'

    temp = np.ones((2, 6), dtype=np.bool)
    assert np.array_equal(region_0, temp)
    temp = np.ones((3, 4), dtype=np.bool)
    temp[0, 0] = 0
    assert np.array_equal(region_1, temp)

    assert np.array_equal(region_0[Axes.Y.value], [0, 1])
    assert np.array_equal(region_0[Axes.X.value], [0, 1, 2, 3, 4, 5])

    assert np.array_equal(region_1[Axes.Y.value], [2, 3, 4])
    assert np.array_equal(region_1[Axes.X.value], [2, 3, 4, 5])
