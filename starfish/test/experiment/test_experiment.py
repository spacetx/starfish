import starfish.data
from starfish.experiment.experiment import Experiment, FieldOfView
from starfish.util.synthesize import SyntheticData


def test_fov_order():
    data = SyntheticData()
    codebook = data.codebook()
    fovs = [FieldOfView("stack2"),
            FieldOfView("stack1")]
    extras = {"synthetic": True}
    experiment = Experiment(fovs, codebook, extras)
    assert "stack1" == experiment.fov().name
    assert ["stack1", "stack2"] == [x.name for x in experiment.fovs()]


def test_crop_experiment():
    exp = starfish.data.ISS(use_test_data=True)
    x_slice = slice(10, 30)
    y_slice = slice(40, 70)
    image = exp['fov_001'].get_image('primary', x_slice=x_slice, y_slice=y_slice)
    assert image.shape['x'] == 20
    assert image.shape['y'] == 30
