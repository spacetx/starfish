import os
import tempfile
import warnings

import numpy as np
import pytest

from starfish import ImageStack
from starfish.recipe import ConstructorError, ConstructorExtraParameterWarning
from starfish.recipe.recipe import Recipe
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


def test_simple_recipe():
    """Test that a simple recipe can execute correctly."""
    recipe_str = """
file_outputs[0] = compute("filter", "SimpleFilterAlgorithm", file_inputs[0], multiplicand=0.5)
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, "output.json")
        recipe = Recipe(recipe_str, [URL], [output_path])

        execution = recipe.execution()
        execution.run_and_save()

        result_stack = ImageStack.from_path_or_url(output_path)
        assert np.allclose(
            BASE_EXPECTED * .5,
            result_stack.xarray[2, 2, 0, 40:50, 40:50]
        )


def test_chained_recipe():
    """Test that a recipe with a complex graph can execute correctly."""
    recipe_str = """
a = compute("filter", "SimpleFilterAlgorithm", file_inputs[0], multiplicand=0.5)
b = compute("filter", "SimpleFilterAlgorithm", a, multiplicand=.3)
file_outputs[0] = compute("filter", "SimpleFilterAlgorithm", b, multiplicand=0.2)
c = compute("filter", "SimpleFilterAlgorithm", a, multiplicand=.2)
file_outputs[1] = compute("filter", "SimpleFilterAlgorithm", c, multiplicand=.3)
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_0_path = os.path.join(tempdir, "output_0.json")
        output_1_path = os.path.join(tempdir, "output_1.json")
        recipe = Recipe(recipe_str, [URL], [output_0_path, output_1_path])

        execution = recipe.execution()
        execution.run_and_save()

        for path in (output_0_path, output_1_path):
            result_stack = ImageStack.from_path_or_url(path)
            assert np.allclose(
                BASE_EXPECTED * .03,
                result_stack.xarray[2, 2, 0, 40:50, 40:50]
            )


def test_garbage_collection():
    """Test that recipe execution discards intermediate results that are no longer necessary."""
    recipe_str = """
a = compute("filter", "SimpleFilterAlgorithm", file_inputs[0], multiplicand=0.5)
b = compute("filter", "SimpleFilterAlgorithm", a, multiplicand=0.3)
c = compute("filter", "SimpleFilterAlgorithm", b, multiplicand=4.)
d = compute("filter", "SimpleFilterAlgorithm", c, multiplicand=0.5)
file_outputs[0] = compute("filter", "SimpleFilterAlgorithm", d, multiplicand=0.5)
    """

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, "output.json")
        recipe = Recipe(recipe_str, [URL], [output_path])

        execution = recipe.execution()
        execution.run_one_tick()
        execution.run_one_tick()

        assert len(execution._completed_results) == 1

        execution.run_and_save()

        result_stack = ImageStack.from_path_or_url(output_path)
        assert np.allclose(
            BASE_EXPECTED * .15,
            result_stack.xarray[2, 2, 0, 40:50, 40:50]
        )


def test_recipe_constructor_missing_args():
    """Test that recipe construction detects missing arguments to the constructor."""
    recipe_str = """
file_output[0] = compute("filter", "SimpleFilterAlgorithm", file_inputs[0])
"""

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, "output.json")
        with pytest.raises(ConstructorError):
            Recipe(recipe_str, [URL], [output_path])


def test_recipe_constructor_extra_args():
    """Test that recipe construction detects missing arguments to the constructor."""
    recipe_str = """
file_outputs[0] = compute("filter", "SimpleFilterAlgorithm", file_inputs[0], multiplicand=.5, x=1)
"""

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = os.path.join(tempdir, "output.json")
        with warnings.catch_warnings(record=True) as w:
            recipe = Recipe(recipe_str, [URL], [output_path])

            assert len(w) == 1
            assert issubclass(w[-1].category, ConstructorExtraParameterWarning)

        execution = recipe.execution()
        execution.run_and_save()

        result_stack = ImageStack.from_path_or_url(output_path)
        assert np.allclose(
            BASE_EXPECTED * .5,
            result_stack.xarray[2, 2, 0, 40:50, 40:50]
        )
