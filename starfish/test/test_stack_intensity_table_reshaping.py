import numpy as np

# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import single_synthetic_spot
from starfish.image import ImageStack


def test_reshaping_between_stack_and_intensities(single_synthetic_spot):
    """
    transform an pixels of an ImageStack into an IntensityTable and back again, then verify that
    the created Imagestack is the same as the original
    """
    codebook, intensities, image = single_synthetic_spot
    pixel_intensities = image.to_pixel_intensities()
    image_from_pixels = ImageStack.from_pixel_intensities(pixel_intensities, assume_contiguous=True)
    assert np.array_equal(image.numpy_array, image_from_pixels.numpy_array)
