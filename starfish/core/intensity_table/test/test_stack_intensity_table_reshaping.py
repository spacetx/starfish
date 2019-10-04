from typing import Tuple

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes
from ..intensity_table import IntensityTable


def test_reshaping_between_stack_and_intensities():
    """
    transform an pixels of an ImageStack into an IntensityTable and back again, then verify that
    the created Imagestack is the same as the original
    """
    np.random.seed(777)
    image = ImageStack.from_numpy(np.random.rand(1, 2, 3, 4, 5).astype(np.float32))
    pixel_intensities = IntensityTable.from_image_stack(image, 0, 0, 0)
    image_shape = (image.shape['z'], image.shape['y'], image.shape['x'])
    image_from_pixels = pixel_intensities_to_imagestack(pixel_intensities, image_shape)
    assert np.array_equal(image.xarray, image_from_pixels.xarray)


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
        intensities.sizes[Axes.ROUND],
        intensities.sizes[Axes.CH]])
    data = data.transpose(3, 4, 0, 1, 2)
    return ImageStack.from_numpy(data)
