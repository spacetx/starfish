import warnings

import numpy as np
import pytest

from starfish import ImageStack
from starfish.recipe import (
    ConstructorError,
    ConstructorExtraParameterWarning,
    ExecutionError,
    RunInsufficientParametersError,
    Runnable,
    TypeInferenceError,
)
from starfish.recipe.filesystem import FileProvider
from . import fakefilter  # noqa: F401


BASE_EXPECTED = np.array([
    [0.227543, 0.223117, 0.217014, 0.221241, 0.212863, 0.211963, 0.210575,
     0.198611, 0.194827, 0.181964],
    [0.216617, 0.214710, 0.212467, 0.218158, 0.211429, 0.210361, 0.205737,
     0.190814, 0.182010, 0.165667],
    [0.206744, 0.204685, 0.208774, 0.212909, 0.215274, 0.206180, 0.196674,
     0.179080, 0.169207, 0.157549],
    [0.190845, 0.197131, 0.188540, 0.195361, 0.196765, 0.200153, 0.183627,
     0.167590, 0.159930, 0.150805],
    [0.181231, 0.187457, 0.182910, 0.179416, 0.175357, 0.172137, 0.165072,
     0.156344, 0.153735, 0.150378],
    [0.169924, 0.184604, 0.182422, 0.174441, 0.159823, 0.157229, 0.157259,
     0.151690, 0.147265, 0.139940],
    [0.164874, 0.169467, 0.178012, 0.173129, 0.161425, 0.155978, 0.152712,
     0.150286, 0.145159, 0.140658],
    [0.164508, 0.165042, 0.171420, 0.174990, 0.162951, 0.152422, 0.149325,
     0.151675, 0.141588, 0.139010],
    [0.162448, 0.156451, 0.158419, 0.162722, 0.160388, 0.152865, 0.142885,
     0.142123, 0.140093, 0.135836],
    [0.150072, 0.147295, 0.145495, 0.153216, 0.156085, 0.149981, 0.145571,
     0.141878, 0.138857, 0.136965]],
    dtype=np.float32)
URL = "https://d2nhj9g34unfro.cloudfront.net/20181005/ISS-TEST/fov_001/hybridization.json"


def test_str():
    """Verify that we can get a sane string for a runnable."""
    filter_runnable = Runnable(
        "filter", "SimpleFilterAlgorithm",
        FileProvider(URL),
        multiplicand=.5,
    )
    assert str(filter_runnable) == ("compute(\"filter\", \"SimpleFilterAlgorithm\","
                                    + f" FileProvider(\"{URL}\"), multiplicand=0.5)")


def test_constructor_error():
    """Verify that we get a properly typed error when the constructor does not execute correctly."""
    with pytest.raises(ConstructorError):
        Runnable(
            "filter", "SimpleFilterAlgorithm",
            FileProvider(URL),
        )


def test_execution_error():
    """Verify that we get a properly typed error when the constructor does not execute correctly."""
    filter_runnable = Runnable(
        "filter", "SimpleFilterAlgorithm",
        FileProvider(URL),
        multiplicand="abacadabra",
    )
    with pytest.raises(ExecutionError):
        filter_runnable.run({})


def test_constructor_type_inference_error():
    """Verify that we get a properly typed error when we cannot properly infer the type for one of
    the constructor's parameters."""
    with pytest.raises(TypeInferenceError):
        Runnable(
            "filter", "FilterAlgorithmWithMissingConstructorTyping",
            FileProvider(URL),
            additive=FileProvider(URL),
        )


def test_run_type_inference_error():
    """Verify that we get a properly typed error when we cannot properly infer the type for one of
    the run method's parameters."""
    with pytest.raises(TypeInferenceError):
        Runnable(
            "filter", "FilterAlgorithmWithMissingRunTyping",
            FileProvider(URL),
            multiplicand=FileProvider(URL),
        )


def test_extra_constructor_parameter_fileprovider():
    """Verify that we raise a warning when we provide extra parameters that are fileproviders to an
    algorithm's constructor."""
    with warnings.catch_warnings(record=True) as w:
        filter_runnable = Runnable(
            "filter", "SimpleFilterAlgorithm",
            FileProvider(URL),
            multiplicand=.5,
            additive=FileProvider(URL),
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, ConstructorExtraParameterWarning)

    result = filter_runnable.run({})
    assert isinstance(result, ImageStack)

    # pick a random part of the filtered image and assert on it
    assert result.xarray.dtype == np.float32

    assert np.allclose(
        BASE_EXPECTED * .5,
        result.xarray[2, 2, 0, 40:50, 40:50]
    )


def test_extra_constructor_parameter_non_fileprovider():
    """Verify that we raise a warning when we provide extra parameters that are not fileproviders
    to an algorithm's constructor."""
    with warnings.catch_warnings(record=True) as w:
        filter_runnable = Runnable(
            "filter", "SimpleFilterAlgorithm",
            FileProvider(URL),
            multiplicand=.5,
            additive=.5,
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, ConstructorExtraParameterWarning)

    result = filter_runnable.run({})
    assert isinstance(result, ImageStack)

    # pick a random part of the filtered image and assert on it
    assert result.xarray.dtype == np.float32

    assert np.allclose(
        BASE_EXPECTED * .5,
        result.xarray[2, 2, 0, 40:50, 40:50]
    )


def test_run_insufficient_parameters():
    """Verify that we can run a single runnable and get its result.
    """
    with pytest.raises(RunInsufficientParametersError):
        Runnable(
            "filter", "SimpleFilterAlgorithm",
            multiplicand=.5,
        )


def test_run():
    """Verify that we can run a single runnable and get its result.
    """
    filter_runnable = Runnable(
        "filter", "SimpleFilterAlgorithm",
        FileProvider(URL),
        multiplicand=.5,
    )
    result = filter_runnable.run({})
    assert isinstance(result, ImageStack)

    # pick a random part of the filtered image and assert on it
    assert result.xarray.dtype == np.float32

    assert np.allclose(
        BASE_EXPECTED * .5,
        result.xarray[2, 2, 0, 40:50, 40:50]
    )


def test_chained_run():
    """Verify that we can run a runnable that depends on another runnable.
    """
    dependency_runnable = Runnable(
        "filter", "SimpleFilterAlgorithm",
        FileProvider(URL),
        multiplicand=.5,
    )
    result = dependency_runnable.run({})
    assert isinstance(result, ImageStack)

    filter_runnable = Runnable(
        "filter", "SimpleFilterAlgorithm",
        dependency_runnable,
        multiplicand=2.0,
    )
    result = filter_runnable.run({dependency_runnable: result})
    assert isinstance(result, ImageStack)

    # pick a random part of the filtered image and assert on it
    assert result.xarray.dtype == np.float32

    assert np.allclose(
        BASE_EXPECTED,
        result.xarray[2, 2, 0, 40:50, 40:50]
    )


def test_chained_run_result_not_present():
    """Verify that we can run a runnable that depends on another runnable, but the results are not
    present.
    """
    dependency_runnable = Runnable(
        "filter", "SimpleFilterAlgorithm",
        FileProvider(URL),
        multiplicand=.5,
    )
    result = dependency_runnable.run({})
    assert isinstance(result, ImageStack)

    filter_runnable = Runnable(
        "filter", "SimpleFilterAlgorithm",
        dependency_runnable,
        multiplicand=2.0,
    )
    with pytest.raises(KeyError):
        filter_runnable.run({})


def test_load_data_for_constructor():
    """Verify that we can properly load up data from a FileProvider that is required for the
    constructor."""
    filter_runnable = Runnable(
        "filter", "AdditiveFilterAlgorithm",
        FileProvider(URL),
        additive=FileProvider(URL),
    )
    result = filter_runnable.run({})
    assert isinstance(result, ImageStack)

    # pick a random part of the filtered image and assert on it
    assert result.xarray.dtype == np.float32

    assert np.allclose(
        BASE_EXPECTED * 2,
        result.xarray[2, 2, 0, 40:50, 40:50]
    )
