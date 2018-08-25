import numpy as np

from starfish.spots._detector.combine_adjacent_features import TargetsMap


def test_targets_map():

    targets = np.array(['None', 'test', 'test_2', 'I_Am_A-Gene', 'I.am.an.R.user'])
    target_map = TargetsMap(targets)

    encoded = target_map.targets_as_int(targets)
    decoded = target_map.targets_as_str(encoded)

    assert np.array_equal(decoded, targets)
