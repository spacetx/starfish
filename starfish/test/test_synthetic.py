import numpy as np

from starfish.util.synthesize import one_hot_code
from starfish.constants import Indices
from starfish.pipeline.features.codebook import Codebook


def test_one_hot_codes():

    n_hyb = 4
    n_channel = 3
    n_codes = 5

    codes = one_hot_code(n_hyb=n_hyb, n_channel=n_channel, n_codes=n_codes)

    assert isinstance(codes, list)

    assert len(codes) == n_codes

    # attempt to load into Codebook object
    cb = Codebook.from_code_array(codes, n_hyb=n_hyb, n_ch=n_channel)

    assert np.array_equal(cb.sum(Indices.CH.value), np.ones_like(cb.sum(Indices.CH.value)))
