import numpy as np
import pytest

from starfish import ImageStack
from starfish.core.spots.FindSpots import BlobDetector
from starfish.core.types import Axes


def test_blob_doh_error_handling():
    """
    Test that BlobDetector throws a Value error if a user tries to use blob_doh on 3d data.
    `skimage.blob_doh` only supports 2d data.
    """
    stack = ImageStack.from_numpy(np.zeros((4, 2, 10, 100, 100), dtype=np.float32))

    blob_doh = BlobDetector(
        min_sigma=1,
        max_sigma=4,
        num_sigma=5,
        threshold=0,
        detector_method='blob_doh',
        measurement_type='max',
        is_volume=True)

    # Check that Value error gets raised when blob_doh and is_volume=True provided
    with pytest.raises(ValueError):
        blob_doh.run(stack)
    ref_image = stack.reduce({Axes.ROUND, Axes.CH}, func='max')
    # Check that Value error gets raised when blob_doh and reference image is 3d
    with pytest.raises(ValueError):
        blob_doh.run(stack, reference_image=ref_image)
