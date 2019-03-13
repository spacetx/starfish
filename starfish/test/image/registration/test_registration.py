from starfish import data
from starfish.image._apply_transform.warp import Warp
from starfish.image._learn_transform.translation import Translation
from starfish.types import Axes


def test_learn_throws_error():
    exp = data.ISS(use_test_data=True)
    stack = exp.fov().get_image('primary')
    reference_stack = exp.fov().get_image('dots')
    translation = Translation(reference_stack=reference_stack)
    try:
        translation.run(stack, Axes.ROUND)
    except ValueError as e:
        # Assert value error is thrown with right message
        assert e.args[0] == "Only axes: r can have a length > 1"


def test_learn_translation():
    exp = data.ISS(use_test_data=True)
    stack = exp.fov().get_image('primary')
    reference_stack = exp.fov().get_image('dots')
    translation = Translation(reference_stack=reference_stack)
    # Calculate max_proh accrss
    stack = stack.max_proj(Axes.CH, Axes.ZPLANE)
    transform_list = translation.run(stack, Axes.ROUND)
    # assert there's a transofrmation object for each round
    assert len(transform_list.transforms) == stack.num_rounds


def test_apply():
    exp = data.ISS(use_test_data=True)
    stack = exp.fov().get_image('primary')
    reference_stack = exp.fov().get_image('dots')
    translation = Translation(reference_stack=reference_stack)
    # Calculate max_proh accrss
    stack = stack.max_proj(Axes.CH, Axes.ZPLANE)
    transform_list = translation.run(stack, Axes.ROUND)
    apply_transform = Warp(transform_list)
    apply_transform.run(stack)
