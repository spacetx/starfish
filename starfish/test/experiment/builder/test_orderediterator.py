from starfish.experiment.builder.orderediterator import ordered_iterator
from starfish.types import Indices


def test_round_then_ch():
    results = list(ordered_iterator(((Indices.ROUND, 3), (Indices.CH, 2))))
    for ix in (0, 1):
        assert results[ix][Indices.ROUND] == 0
    for ix in (2, 3):
        assert results[ix][Indices.ROUND] == 1
    for ix in (4, 5):
        assert results[ix][Indices.ROUND] == 2

    for ix in (0, 2, 4):
        assert results[ix][Indices.CH] == 0
    for ix in (1, 3, 5):
        assert results[ix][Indices.CH] == 1


def test_ch_then_round():
    results = list(ordered_iterator(((Indices.CH, 2), (Indices.ROUND, 3))))
    for ix in (0, 3):
        assert results[ix][Indices.ROUND] == 0
    for ix in (1, 4):
        assert results[ix][Indices.ROUND] == 1
    for ix in (2, 5):
        assert results[ix][Indices.ROUND] == 2

    for ix in (0, 1, 2):
        assert results[ix][Indices.CH] == 0
    for ix in (3, 4, 5):
        assert results[ix][Indices.CH] == 1
