"""
This module stores a contract for new Filter algorithms. 

Contract
--------

- constructor accepts 3d and 2d data, where default is 2d
- values emitted by a filter are floats between 0 and 1 (inclusive)
- exposes a `run`() method
- run accepts an in-place parameter which defaults to True
- run always returns an ImageStack (if in-place, returns a reference to the modified input data)
- run accepts an `n_processes` parameter which determines 
- run accepts a `verbose` parameter, which triggers tqdm to print progress

To add a new filter, simply add default
parameters for the constructor (omit is_volume), and it will be tested against the contract by
way of registration to the FilterAlgorithmBase
"""

import numpy as np
import pytest

from starfish import ImageStack
from starfish.image import Filter

methods = Filter.implementing_algorithms()

# store some default parameters for testing purposes. New algorithms will have to add parameters to
# this in order to not fail tests
default_filtering_parameters = {
    'Bandpass': dict(lshort=1, llong=3, threshold=0.01),
    'ZeroByChannelMagnitude': dict(thresh=0, normalize=True),
    'WhiteTophat': dict(masking_radius=3),
    'ScaleByPercentile': dict(p=0),
    'DeconvolvePSF': dict(num_iter=1, sigma=1),
    'MeanHighPass': dict(size=1),
    'GaussianLowPass': dict(sigma=1),
    'GaussianHighPass': dict(sigma=3),
    'Clip': dict(p_min=0, p_max=100),
}


def generate_default_data():
    data = np.random.rand(2, 2, 2, 40, 50).astype(np.float32)
    return ImageStack.from_numpy_array(data)


@pytest.mark.parametrize('filter_class', methods)
def test_all_methods_adhere_to_contract(filter_class):
    """Test that all filter algorithms adhere to the filtering contract"""

    default_kwargs = default_filtering_parameters[filter_class.get_algorithm_name()]

    # accept boolean is_volume
    instance = filter_class(is_volume=True, **default_kwargs)

    # TODO ambrosejcarr: all methods should process 3d data
    # # stores is_volume, and is_volume is a bool
    # try:
    #     volume_param = getattr(instance, 'is_volume')
    # except AttributeError:
    #     raise AttributeError(f'{filter_class} should accept and store is_volume.')
    #
    # assert isinstance(volume_param, bool), \
    #     f'{filter_class} is_volume must be a bool, not {type(volume_param)}'

    # always emits an Image, even if in_place=True and the resulting filter operates in-place
    data = generate_default_data()
    try:
        filtered = instance.run(data, in_place=True)
    except TypeError as e:
        raise AssertionError(f'{filter_class} must accept in_place parameter')
    assert isinstance(filtered, ImageStack)
    assert data is filtered, \
        f'{filter_class} should return a reference to the input ImageStack when run in_place'

    # operates out of place
    data = generate_default_data()
    filtered = instance.run(data, in_place=False)
    assert data is not filtered, \
        f'{filter_class} should output a new ImageStack when run out-of-place'

    # accepts n_processes
    data = generate_default_data()
    try:
        instance.run(data, n_processes=1)
    except TypeError as e:
        raise AssertionError(f'{filter_class} must accept n_processes parameter')

    # accepts verbose, and if passed, prints progress
    data = generate_default_data()
    try:
        instance.run(data, verbose=True)
    except TypeError as e:
        raise AssertionError(f'{filter_class} must accept verbose parameter')

    # output is dtype float and within the expected interval of [0, 1]
    assert filtered.numpy_array.dtype == np.float32, f'{filter_class} must output float32 data'
    assert np.all(filtered.numpy_array >= 0), \
        f'{filter_class} must output a result where all values are >= 0'
    assert np.all(filtered.numpy_array <= 1), \
        f'{filter_class} must output a result where all values are <= 1'
