import starfish.data
from starfish.experiment.experiment import Experiment, FieldOfView
from starfish.imagestack.imagestack import ImageStack
from starfish.imagestack.parser.crop import CropParameters
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


def test_crop_experiment():
    exp = starfish.data.ISS(use_test_data=True)
    c = CropParameters(x_slice=slice(10, 30), y_slice=slice(40, 70))
    image = exp['fov_001'].get_image('primary', c)
    assert image.shape['x'] == 20
    assert image.shape['y'] == 30
