import numpy as np

from starfish import ImageStack
from starfish.core.image.Filter.zero_by_channel_magnitude import ZeroByChannelMagnitude

def create_imagestack_with_magnitude_scale():
    """create an imagestack with increasing magnitudes"""
    data = np.linspace(0, 1, 11, dtype=np.float32)
    data = np.repeat(data[None, :], 2, axis=0)
    # reshape data into a 2-channel, (1, 11, 1) image in (x, y, z)
    data = data.reshape(1, 2, 1, 11, 1)
    imagestack = ImageStack.from_numpy(data)
    return imagestack


def test_zero_by_channel_magnitude_produces_accurate_results():
    imagestack = create_imagestack_with_magnitude_scale()

    zcm = ZeroByChannelMagnitude(thresh=np.inf, normalize=False)
    filtered = zcm.run(imagestack, in_place=False, n_processes=1)
    assert np.all(filtered.xarray == 0)
