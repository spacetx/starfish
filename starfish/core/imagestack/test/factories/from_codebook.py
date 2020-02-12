from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from starfish.core.codebook.codebook import Codebook
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, Features


def create_imagestack_from_codebook(
    pixel_dimensions: Tuple[int, int, int],
    spot_coordinates: Sequence[Tuple[int, int, int]],
    codebook: Codebook
) -> ImageStack:
    """
    creates a numpy array containing one spot per codebook entry at spot_coordinates. length of
    spot_coordinates must therefore match the number of codes in Codebook.
    """
    assert len(spot_coordinates) == codebook.sizes[Features.TARGET]

    data_shape = (
        codebook.sizes[Axes.ROUND.value],
        codebook.sizes[Axes.CH.value],
        *pixel_dimensions
    )
    imagestack_data = np.zeros(data_shape, dtype=np.float32)

    for ((z, y, x), f) in zip(spot_coordinates, range(codebook.sizes[Features.TARGET])):
        imagestack_data[:, :, z, y, x] = codebook[f].transpose(Axes.ROUND.value, Axes.CH.value)

    # blur with a small non-isotropic kernel
    imagestack_data = gaussian_filter(imagestack_data, sigma=(0, 0, 0.7, 1.5, 1.5))
    return ImageStack.from_numpy(imagestack_data)
