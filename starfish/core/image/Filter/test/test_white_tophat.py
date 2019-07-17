import numpy as np
import pytest
from skimage.filters import gaussian

from starfish.core.image.Filter.white_tophat import WhiteTophat


def simple_spot_3d():

    big_spot = np.zeros((100, 100, 100), dtype=np.uint16)
    big_spot[20, 20, 20] = 10000
    big_spot = gaussian(big_spot, sigma=(4, 4, 4), preserve_range=True).astype(np.uint16)

    small_spot = np.zeros((100, 100, 100), dtype=np.uint16)
    small_spot[80, 80, 80] = 10000
    small_spot = gaussian(small_spot, sigma=(1, 1, 1), preserve_range=True).astype(np.uint16)

    return big_spot + small_spot


@pytest.mark.parametrize('is_volume', [True, False])
def test_white_tophat(is_volume: bool):
    """
    white tophat filters should reduce the intensity of spots that are larger than the masking
    radius much more than those that are smaller. Here we have a 4-diameter spot and a 2-diameter
    spot. We expect the large spot to be filtered more substantially.

    Verify this functionality in both 2d and 3d images
    """
    if is_volume:

        image = simple_spot_3d()
        small_spot_intensity = image[80, 80, 80]
        big_spot_intensity = image[20, 20, 20]

        wth = WhiteTophat(masking_radius=2, is_volume=True)
        filtered = wth._white_tophat(image)

        large_ratio = filtered[20, 20, 20] / big_spot_intensity
        small_ratio = filtered[80, 80, 80] / small_spot_intensity

    else:

        image = simple_spot_3d().max(axis=0)
        small_spot_intensity = image[80, 80]
        big_spot_intensity = image[20, 20]

        wth = WhiteTophat(masking_radius=2, is_volume=False)
        filtered = wth._white_tophat(image)

        large_ratio = filtered[20, 20] / big_spot_intensity
        small_ratio = filtered[80, 80] / small_spot_intensity

    assert large_ratio < small_ratio
