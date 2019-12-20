from starfish import data
from starfish.image import Segment


def test_from_illastik():
    merfish = data.MERFISH()
    fov = merfish['fov_000']

    segmenter = Segment.IllastikPretrained('/Users/shannonaxelrod/dev/starfish/notebooks/dapi_Probabilities.h5')
    dapi = fov.get_image("nuclei")
    primary_image = fov.get_image('primary')

    binary_masks = segmenter.run(primary_image, dapi)
    return binary_masks
