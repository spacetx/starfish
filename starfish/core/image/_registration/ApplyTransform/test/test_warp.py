import numpy as np

from starfish import data
from starfish.core.image import Filter
from starfish.core.image._registration.ApplyTransform.warp import Warp
from starfish.core.image._registration.LearnTransform.translation import Translation
from starfish.core.types import Axes


expected_registered_values = np.array(
    [[0.090654, 0.090593, 0.091554, 0.091661, 0.089967, 0.094072, 0.097398,
      0.099046, 0.100969, 0.112108],
     [0.09926, 0.096925, 0.096269, 0.097002, 0.095842, 0.097704, 0.09984,
      0.101457, 0.105455, 0.106004],
     [0.109834, 0.103609, 0.102693, 0.099931, 0.098222, 0.10074, 0.10251,
      0.103838, 0.106874, 0.113451],
     [0.12369, 0.112428, 0.111482, 0.10631, 0.106203, 0.104753, 0.106706,
      0.105013, 0.10811, 0.11371],
     [0.141802, 0.129946, 0.124285, 0.120928, 0.115908, 0.110735, 0.110735,
      0.107454, 0.109468, 0.109255],
     [0.147326, 0.14464, 0.141436, 0.132845, 0.124071, 0.121828, 0.118074,
      0.112306, 0.109163, 0.109483],
     [0.145296, 0.150362, 0.15082, 0.140337, 0.133806, 0.1299, 0.120592,
      0.114046, 0.115496, 0.111666],
     [0.131121, 0.145525, 0.150011, 0.146609, 0.137407, 0.129198, 0.127306,
      0.118029, 0.116594, 0.111559],
     [0.126482, 0.132372, 0.142596, 0.149538, 0.144701, 0.137469, 0.125353,
      0.121996, 0.117342, 0.118273],
     [0.122866, 0.126543, 0.133669, 0.145418, 0.150515, 0.140139, 0.129992,
      0.124605, 0.120867, 0.121889]], dtype=np.float32)


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
