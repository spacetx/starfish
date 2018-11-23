from starfish.experiment.experiment import Experiment, FieldOfView
from starfish.imagestack.imagestack import ImageStack
from starfish.util.synthesize import SyntheticData


def test_fov_order():
    data = SyntheticData()
    codebook = data.codebook()
    stack1 = ImageStack.synthetic_stack(num_round=5, num_ch=5, num_z=15,
                                        tile_height=200, tile_width=200)
    stack2 = ImageStack.synthetic_stack(num_round=5, num_ch=5, num_z=15,
                                        tile_height=200, tile_width=200)
    fovs = [FieldOfView("stack2", {"primary": stack2}),
            FieldOfView("stack1", {"primary": stack1})]
    extras = {"synthetic": True}
    experiment = Experiment(fovs, codebook, extras)
    assert "stack1" == experiment.fov().name
    assert ["stack1", "stack2"] == [x.name for x in experiment.fovs()]
