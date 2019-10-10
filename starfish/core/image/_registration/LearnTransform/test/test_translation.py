import numpy as np

from starfish import data
from starfish.core.image import Filter
from starfish.core.image._registration.LearnTransform.translation import Translation
from starfish.core.types import Axes


ISS_SHIFTS = [[-23, 6], [-22, 2], [-22, -3], [-15, -4]]


def test_learn_transforms_throws_error():
    exp = data.ISS(use_test_data=True)
    stack = exp.fov().get_image('primary')
    reference_stack = exp.fov().get_image('dots')
    translation = Translation(reference_stack=reference_stack, axes=Axes.ROUND)
    try:
        translation.run(stack)
    except ValueError as e:
        # Assert value error is thrown when the stack is not max projected across all other axes.
        assert e.args[0] == "Only axes: r can have a length > 1, please use the MaxProj filter."


def test_learn_transforms_translation():
    exp = data.ISS(use_test_data=True)
    stack = exp.fov().get_image('primary')
    reference_stack = exp.fov().get_image('dots')
    translation = Translation(reference_stack=reference_stack, axes=Axes.ROUND)
    # Calculate max_proj accrss CH/Z
    stack = Filter.Reduce((Axes.CH, Axes.ZPLANE)).run(stack)
    transform_list = translation.run(stack)
    # assert there's a transofrmation object for each round
    assert len(transform_list.transforms) == stack.num_rounds
    for (_, _, transform), shift in zip(transform_list.transforms, ISS_SHIFTS):
        # assert that each TransformationObject has the correct translation shift
        assert np.array_equal(transform.translation, shift)
