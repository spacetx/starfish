import numpy as np
import pytest

from starfish import BinaryMaskCollection, display, LabelImage
from starfish.core.test.factories import SyntheticData
from starfish.types import Coordinates


sd = SyntheticData(
    n_ch=2,
    n_round=3,
    n_spots=1,
    n_codes=4,
    n_photons_background=0,
    background_electrons=0,
    camera_detection_efficiency=1.0,
    gray_level=1,
    ad_conversion_bits=16,
    point_spread_function=(2, 2, 2),
)

stack = sd.spots()
spots = sd.intensities()
label_image = LabelImage.from_label_array_and_ticks(
    np.random.randint(0, 4, size=(128, 128), dtype=np.uint8),
    None,
    {Coordinates.Y: np.arange(128), Coordinates.X: np.arange(128)},
    None,
)
masks = BinaryMaskCollection.from_label_image(label_image)


@pytest.mark.napari
@pytest.mark.parametrize('masks', [masks, None], ids=['masks', '     '])
@pytest.mark.parametrize('spots', [spots, None], ids=['spots', '     '])
@pytest.mark.parametrize('stack', [stack, None], ids=['stack', '     '])
def test_display(qtbot, stack, spots, masks):
    from napari import Viewer

    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    if stack is None and spots is None and masks is None:
        with pytest.raises(TypeError):
            display(stack, spots, masks, viewer=viewer)
    else:
        display(stack, spots, masks, viewer=viewer)
