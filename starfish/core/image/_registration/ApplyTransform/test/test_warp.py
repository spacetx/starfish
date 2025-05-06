import numpy as np

from starfish import data
from starfish.core.image import Filter
from starfish.core.image._registration.ApplyTransform.warp import Warp
from starfish.core.image._registration.LearnTransform.translation import Translation
from starfish.core.types import Axes

expected_registered_values = np.array(
    [[0.09065385, 0.09059282, 0.09155414, 0.09166095, 0.08996719,
      0.09407187, 0.09739833, 0.09904631, 0.10096895, 0.11210804],
     [0.09925994, 0.0969253, 0.09626917, 0.0970016, 0.09584191,
      0.09770352, 0.09983978, 0.10145724, 0.1054551, 0.10600442],
     [0.10983444, 0.10360876, 0.10269322, 0.09993134, 0.09822232,
      0.10074006, 0.10251011, 0.10383765, 0.10687419, 0.11345083],
     [0.12368963, 0.11242847, 0.11148241, 0.10630961, 0.1062028,
      0.10475319, 0.10670634, 0.10501259, 0.10811017, 0.11371023],
     [0.14180209, 0.12994583, 0.12428474, 0.12092775, 0.11590753,
      0.11073472, 0.11073472, 0.10745403, 0.10946822, 0.1092546],
     [0.14732586, 0.14464027, 0.14143588, 0.13284504, 0.12407111,
      0.12182803, 0.11807431, 0.1123064, 0.10916305, 0.10948348],
     [0.14529641, 0.1503624, 0.15082017, 0.14033723, 0.13380636,
      0.12990005, 0.12059205, 0.11404593, 0.11549554, 0.11166552],
     [0.13112077, 0.14552529, 0.15001145, 0.14660868, 0.1374075,
      0.12919813, 0.12730601, 0.11802854, 0.11659419, 0.11155871],
     [0.12648204, 0.13237202, 0.14259556, 0.14953841, 0.1447013,
      0.13746853, 0.12535286, 0.12199588, 0.11734188, 0.11827268],
     [0.12286565, 0.12654307, 0.13366903, 0.14541848, 0.15051499,
      0.14013886, 0.1299916 , 0.12460517, 0.12086672, 0.12188907]], dtype=np.float32)

expected_registered_values_with_phase_normalization = np.array(
    [[0.085832, 0.085084, 0.086229, 0.08687, 0.089662, 0.092256,
      0.099474, 0.099489, 0.11017, 0.122408],
     [0.090654, 0.090593, 0.091554, 0.091661, 0.089967, 0.094072,
      0.097398, 0.099046, 0.100969, 0.112108],
     [0.09926, 0.096925, 0.096269, 0.097002, 0.095842, 0.097704,
      0.09984, 0.101457, 0.105455, 0.106004],
     [0.109834, 0.103609, 0.102693, 0.099931, 0.098222, 0.10074,
      0.10251, 0.103838, 0.106874, 0.113451],
     [0.12369, 0.112428, 0.111482, 0.10631, 0.106203, 0.104753,
      0.106706, 0.105013, 0.10811, 0.11371],
     [0.141802, 0.129946, 0.124285, 0.120928, 0.115908, 0.110735,
      0.110735, 0.107454, 0.109468, 0.109255],
     [0.147326, 0.14464, 0.141436, 0.132845, 0.124071, 0.121828,
      0.118074, 0.112306, 0.109163, 0.109483],
     [0.145296, 0.150362, 0.15082, 0.140337, 0.133806, 0.1299,
      0.120592, 0.114046, 0.115496, 0.111666],
     [0.131121, 0.145525, 0.150011, 0.146609, 0.137408, 0.129198,
      0.127306, 0.118029, 0.116594, 0.111559],
     [0.126482, 0.132372, 0.142596, 0.149538, 0.144701, 0.137469,
      0.125353, 0.121996, 0.117342, 0.118273]], dtype=np.float32)


def test_calculate_translation_transforms_and_apply():
    exp = data.ISS(use_test_data=True)
    stack = exp.fov().get_image('primary')
    reference_stack = exp.fov().get_image('dots')
    translation = Translation(reference_stack=reference_stack, axes=Axes.ROUND)
    # Calculate max_proj accrss
    mp = Filter.Reduce((Axes.CH, Axes.ZPLANE)).run(stack)
    transform_list = translation.run(mp)
    apply_transform = Warp()
    warped_stack = apply_transform.run(stack=stack, transforms_list=transform_list)
    assert np.allclose(
        expected_registered_values,
        warped_stack.xarray[2, 2, 0, 40:50, 40:50])

def test_calculate_translation_transforms_and_apply_with_phase_normalization():
    exp = data.ISS(use_test_data=True)
    stack = exp.fov().get_image('primary')
    reference_stack = exp.fov().get_image('dots')
    translation = Translation(reference_stack=reference_stack, axes=Axes.ROUND,
                              normalization="phase")
    mp = Filter.Reduce((Axes.CH, Axes.ZPLANE)).run(stack)
    transform_list = translation.run(mp)
    apply_transform = Warp()
    warped_stack = apply_transform.run(stack=stack, transforms_list=transform_list)
    assert np.allclose(
        expected_registered_values_with_phase_normalization,
        warped_stack.xarray[2, 2, 0, 40:50, 40:50])
