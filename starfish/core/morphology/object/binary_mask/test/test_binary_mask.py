import numpy as np

from starfish.core.morphology.object.label_image.label_image import LabelImage
from starfish.core.types import Axes, Coordinates
from ..binary_mask import BinaryMaskCollection


def test_from_label_image():
    label_image_array = np.zeros((5, 5), dtype=np.int32)
    label_image_array[0] = 1
    label_image_array[3:5, 3:5] = 2
    label_image_array[-1, -1] = 0

    physical_ticks = {Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
                      Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12]}

    label_image = LabelImage.from_array_and_coords(
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

    assert np.array_equal(region_0, np.ones((1, 5), dtype=np.bool))
    temp = np.ones((2, 2), dtype=np.bool)
    temp[-1, -1] = False
    assert np.array_equal(region_1, temp)

    assert np.array_equal(region_0[Axes.Y.value], [0])
    assert np.array_equal(region_0[Axes.X.value], [0, 1, 2, 3, 4])

    assert np.array_equal(region_1[Axes.Y.value], [3, 4])
    assert np.array_equal(region_1[Axes.X.value], [3, 4])

    assert np.array_equal(region_0[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][0:1])
    assert np.array_equal(region_0[Coordinates.X.value],
                          physical_ticks[Coordinates.X][0:5])

    assert np.array_equal(region_1[Coordinates.Y.value],
                          physical_ticks[Coordinates.Y][3:5])
    assert np.array_equal(region_1[Coordinates.X.value],
                          physical_ticks[Coordinates.X][3:5])


def test_to_label_image():
    # test via roundtrip
    label_image_array = np.zeros((5, 6), dtype=np.int32)
    label_image_array[0] = 1
    label_image_array[3:6, 3:6] = 2
    label_image_array[-1, -1] = 0

    physical_ticks = {Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
                      Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12, 15.5]}

    label_image = LabelImage.from_array_and_coords(
        label_image_array,
        None,
        physical_ticks,
        None,
    )

    masks = BinaryMaskCollection.from_label_image(label_image)

    assert np.array_equal(masks.to_label_image().xarray, label_image.xarray)


def test_save_load(tmp_path):
    label_image_array = np.zeros((5, 5), dtype=np.int32)
    label_image_array[0] = 1
    label_image_array[3:5, 3:5] = 2
    label_image_array[-1, -1] = 0

    physical_ticks = {Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
                      Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12]}

    label_image = LabelImage.from_array_and_coords(
        label_image_array,
        None,
        physical_ticks,
        None,
    )

    binary_mask_collection = BinaryMaskCollection.from_label_image(label_image)

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
    label_image_array = np.zeros((5, 6), dtype=np.int32)

    physical_ticks = {Coordinates.Y: [1.2, 2.4, 3.6, 4.8, 6.0],
                      Coordinates.X: [7.2, 8.4, 9.6, 10.8, 12, 13.2]}

    label_image = LabelImage.from_array_and_coords(
        label_image_array,
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
