"""
Tests for starfish.spots._detect_spots.combine_adjacent_features.TargetsMap
"""

import numpy as np

from starfish.core.spots.DetectPixels.combine_adjacent_features import TargetsMap


def test_targets_map():
    """Test that TargetsMap provides an invertible map between string names and integer IDs"""

    targets = np.array(['None', 'test', 'test_2', 'I_Am_A-Gene', 'I.am.an.R.user'])
    target_map = TargetsMap(targets)

    encoded = target_map.targets_as_int(targets)
    decoded = target_map.targets_as_str(encoded)

    assert np.array_equal(decoded, targets)
