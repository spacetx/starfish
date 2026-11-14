from copy import deepcopy

import pytest

from starfish.core.spots.DecodeSpots.check_all_funcs import barcodeBuildFunc


@pytest.mark.parametrize("roundOmitNum", [0, 1])
def test_barcode_build_func_does_not_modify_input(roundOmitNum: int) -> None:
    neighbors = {0: [1], 1: [2, 3], 2: [4]}
    original = deepcopy(neighbors)

    barcodeBuildFunc([neighbors], currentRound=0, roundOmitNum=roundOmitNum, roundNum=3)

    assert neighbors == original


def test_barcode_build_func_repeated_input_gives_same_result() -> None:
    neighbors = {0: [1], 1: [2, 3], 2: [4]}

    result = barcodeBuildFunc([neighbors, neighbors], currentRound=0, roundOmitNum=1, roundNum=3)

    assert len(result[0]) == len(result[1])
    assert all((a == b).all() if hasattr(a, "all") else a == b
               for a, b in zip(result[0], result[1]))
