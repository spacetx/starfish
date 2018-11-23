from starfish.experiment.experiment import Experiment, FieldOfView
from starfish.imagestack.imagestack import ImageStack
from starfish.util.synthesize import SyntheticData


def test_fov_order():
    data = SyntheticData()
    codebook = data.codebook()
    stack1 = ImageStack.synthetic_stack()
    stack2 = ImageStack.synthetic_stack()
    fovs = [FieldOfView("stack2", {"primary": stack2}),
            FieldOfView("stack1", {"primary": stack1})]
    extras = {"synthetic": True}
    experiment = Experiment(fovs, codebook, extras)
    assert "stack1" == experiment.fov().name
    assert ["stack1", "stack2"] == [x.name for x in experiment.fovs()]
