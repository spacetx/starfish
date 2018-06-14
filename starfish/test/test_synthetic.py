import numpy as np

from starfish.image import ImageStack
from starfish.util import synthesize
from starfish.constants import Indices
from starfish.pipeline.features.codebook import Codebook


def test_one_hot_codes():

    n_hyb = 4
    n_channel = 3
    n_codes = 5

    codes = Codebook.synthetic_one_hot_codes(n_hyb=n_hyb, n_channel=n_channel, n_codes=n_codes)

    assert isinstance(codes, Codebook)

    assert len(codes) == n_codes

    assert np.array_equal(codes.sum(Indices.CH.value), np.ones_like(codes.sum(Indices.CH.value)))


def test_create_spots():
    codes = Codebook.synthetic_one_hot_codes(4, 2, 10)
    _ = synthesize.create_spots(codes)


def test_synthetic_data():
    stp = synthesize.SyntheticSpotTileProvider()
    ImageStack.synthetic_stack(tile_data_provider=stp.tile)
