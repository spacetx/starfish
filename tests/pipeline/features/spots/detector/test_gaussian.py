import os

from starfish.io import Stack
from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.util.test import DATA_DIR


def test_gaussian_spot_detector():

    # make a stack
    s = Stack()
    s.read(os.path.join(DATA_DIR, 'iss', 'filtered', 'experiment.json'))
    print(s.image.numpy_array.shape)

    gsd = GaussianSpotDetector(stack=s)
    gsd.detect(
        min_sigma=4,
        max_sigma=6,
        num_sigma=20,
        threshold=0.01,
        blobs='dots',
        measurement_type='max',
        bit_map_flag=False
    )

    # TODO ambrosejcarr what's the ground truth here?
