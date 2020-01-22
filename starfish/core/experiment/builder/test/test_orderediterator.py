from starfish.core.types import Axes
from ..orderediterator import ordered_iterator


def test_round_then_ch():
    results = list(ordered_iterator(((Axes.ROUND, range(3)), (Axes.CH, range(2)))))
    for ix in (0, 1):
        assert results[ix][Axes.ROUND] == 0
    for ix in (2, 3):
        assert results[ix][Axes.ROUND] == 1
    for ix in (4, 5):
        assert results[ix][Axes.ROUND] == 2

    for ix in (0, 2, 4):
        assert results[ix][Axes.CH] == 0
    for ix in (1, 3, 5):
        assert results[ix][Axes.CH] == 1


def test_ch_then_round():
    results = list(ordered_iterator(((Axes.CH, range(2)), (Axes.ROUND, range(3)))))
    for ix in (0, 3):
        assert results[ix][Axes.ROUND] == 0
    for ix in (1, 4):
        assert results[ix][Axes.ROUND] == 1
    for ix in (2, 5):
        assert results[ix][Axes.ROUND] == 2

    for ix in (0, 1, 2):
        assert results[ix][Axes.CH] == 0
    for ix in (3, 4, 5):
        assert results[ix][Axes.CH] == 1
