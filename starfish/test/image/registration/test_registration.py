from starfish import data
from starfish.image._apply_transform.warp import Warp
from starfish.image._learn_transform.translation import Translation


def test_learn_and_apply_translation():
    exp = data.ISS(use_test_data=True)
    stack = exp.fov().get_image('primary')
    reference_stack = exp.fov().get_image('dots')
    translation = Translation(reference_stack=reference_stack)
    transformObeject = translation.run(stack)

    # applyTransform = Warp(transformation_object=transformObeject)
    #
    # applyTransform.run(stack)
