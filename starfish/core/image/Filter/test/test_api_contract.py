"""
This module stores a contract for new Filter algorithms.

Contract
--------

- constructor accepts 3d and 2d data, where default is 2d
- values emitted by a filter are floats between 0 and 1 (inclusive)
- exposes a `run`() method
- run always returns an ImageStack (if in-place, returns a reference to the modified input data)
- run accepts a `verbose` parameter, which triggers tqdm to print progress

To add a new filter, simply add default
parameters for the constructor (omit is_volume), and it will be tested against the contract by
way of registration to the FilterAlgorithmBase
"""

from typing import Mapping, Type

import numpy as np
import pytest

from starfish import ImageStack
from starfish.core.image import Filter
from starfish.core.image.Filter.reduce import Reduce


methods: Mapping[str, Type] = {
    'clip': Filter.Clip,
    'bandpass': Filter.Bandpass,
    'clip_percentile_to_zero': Filter.ClipPercentileToZero,
    'clip_value_to_zero': Filter.ClipPercentileToZero,
    'element_wise_mult': Filter.ElementWiseMultiply,
    'guassian_high_pass': Filter.GaussianHighPass,
    'guassian_low_pass': Filter.GaussianLowPass,
    'laplace': Filter.Laplace,
    'mean_high_pass': Filter.MeanHighPass,
    'reduce': Filter.Reduce,
    'deconvolve': Filter.DeconvolvePSF,
    'white_top_hat': Filter.WhiteTophat,
    'zero_by_channel': Filter.ZeroByChannelMagnitude
}


def generate_default_data():
    data = np.random.rand(2, 2, 2, 40, 50).astype(np.float32)
    return ImageStack.from_numpy(data)


@pytest.mark.parametrize('filter_class', methods.values())
def test_all_methods_adhere_to_contract(filter_class):
    """Test that all filter algorithms adhere to the filtering contract"""

    default_kwargs = filter_class._DEFAULT_TESTING_PARAMETERS

    instance = filter_class(**default_kwargs)

    data = generate_default_data()

    # Reduce don't have an in_place, n_processes, verbose option,
    # so we need to skip these tests
    if filter_class not in (Reduce,):
        # return None if in_place = True
        try:
            filtered = instance.run(data, in_place=True)
        except TypeError:
            raise AssertionError(f'{filter_class} must accept in_place parameter')
        assert not filtered

        # operates out of place
        data = generate_default_data()
        filtered = instance.run(data, in_place=False)
        assert data is not filtered, \
            f'{filter_class} should output a new ImageStack when run out-of-place'

        data = generate_default_data()
        try:
            instance.run(data, n_processes=1)
        except TypeError:
            raise AssertionError(f'{filter_class} must accept n_processes parameter')

        # accepts verbose, and if passed, prints progress
        data = generate_default_data()
        try:
            instance.run(data, verbose=True)
        except TypeError:
            raise AssertionError(f'{filter_class} must accept verbose parameter')
    else:
        filtered = instance.run(data)

    # output is dtype float and within the expected interval of [0, 1]
    assert filtered.xarray.dtype == np.float32, f'{filter_class} must output float32 data'
    assert np.all(filtered.xarray >= 0), \
        f'{filter_class} must output a result where all values are >= 0'
    assert np.all(filtered.xarray <= 1), \
        f'{filter_class} must output a result where all values are <= 1'
