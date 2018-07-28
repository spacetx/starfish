import numpy as np
from typing import Tuple

from starfish.constants import Indices
from starfish.image import ImageStack
from starfish.intensity_table import IntensityTable
# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import single_synthetic_spot


def test_reshaping_between_stack_and_intensities(single_synthetic_spot):
    """
    transform an pixels of an ImageStack into an IntensityTable and back again, then verify that
    the created Imagestack is the same as the original
    """
    codebook, intensities, image = single_synthetic_spot
    pixel_intensities = IntensityTable.from_image_stack(image, 0, 0, 0)
    image_shape = (image.shape['z'], image.shape['y'], image.shape['x'])
    image_from_pixels = pixel_intensities_to_imagestack(pixel_intensities, image_shape)
    assert np.array_equal(image.numpy_array, image_from_pixels.numpy_array)


def pixel_intensities_to_imagestack(
        intensities: IntensityTable, image_shape: Tuple[int, int, int]
) -> ImageStack:
    """Re-create the pixel intensities from an IntensityTable

    Parameters
    ----------
    intensities : IntensityTable
        intensities to transform into an ImageStack
    image_shape : Tuple[int, int, int]
        the dimensions of z, y, and x for the original image that the intensity table was generated
        from

    Returns
    -------
    ImageStack :
        ImageStack containing Intensity information

    """
    # reverses the process used to produce the intensity table in to_pixel_intensities
    data = intensities.values.reshape([
        *image_shape,
        intensities.sizes[Indices.CH],
        intensities.sizes[Indices.ROUND]])
    data = data.transpose(4, 3, 0, 1, 2)
    return ImageStack.from_numpy_array(data)
