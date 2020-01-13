import pytest

from starfish.core.morphology.binary_mask.test.factories import binary_mask_collection_2d
from ..areafilter import AreaFilter


def test_empty_filter():
    input_mask_collection = binary_mask_collection_2d()
    output_mask_collection = AreaFilter().run(input_mask_collection)
    assert len(output_mask_collection) == len(input_mask_collection)

    for mask_num in range(len(input_mask_collection)):
        input_mask = input_mask_collection.uncropped_mask(mask_num)
        output_mask = output_mask_collection.uncropped_mask(mask_num)
        assert input_mask.equals(output_mask)


def test_min_area():
    input_mask_collection = binary_mask_collection_2d()
    output_mask_collection = AreaFilter(min_area=6).run(input_mask_collection)
    assert len(output_mask_collection) == 1


def test_max_area():
    input_mask_collection = binary_mask_collection_2d()
    output_mask_collection = AreaFilter(max_area=5).run(input_mask_collection)
    assert len(output_mask_collection) == 1


def test_illegal_areas():
    with pytest.raises(ValueError):
        AreaFilter(min_area=7, max_area=1)
