import numpy as np

from starfish.core.image.Filter.match_histograms import MatchHistograms
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes


def test_match_histograms():
    linear_gradient = np.linspace(0, 0.5, 2000, dtype=np.float32)
    image = linear_gradient.reshape(2, 4, 5, 5, 10)

    # linear_gradient = np.linspace(0, 1, 10)[::-1]
    # grad = np.repeat(linear_gradient[np.newaxis, :], 10, axis=0)
    # image2 = np.tile(grad, (1, 2, 2, 10, 10))

    # because of how the image was structured, every volume should be the same after
    # quantile normalization
    stack = ImageStack.from_numpy(image)
    mh = MatchHistograms({Axes.CH, Axes.ROUND})
    results = mh.run(stack)
    assert len(np.unique(results.xarray.sum(("x", "y", "z")))) == 1

    # because here we are allowing variation to persist across rounds, each
    # round within each channel should be different
    mh = MatchHistograms({Axes.CH})
    results2 = mh.run(stack)
    assert len(np.unique(results2.xarray.sum(("x", "y", "z")))) == 2

    # same as above, but verifying this functions for a different data shape (2 rounds, 4 channels)
    mh = MatchHistograms({Axes.ROUND})
    results2 = mh.run(stack)
    assert len(np.unique(results2.xarray.sum(("x", "y", "z")))) == 4
