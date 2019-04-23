from itertools import product
from typing import Iterator, Mapping, Sequence, Tuple

from starfish.core.types import Axes


def join_axes_labels(
        axes_order: Sequence[Axes],
        *,
        rounds: Sequence[int],
        chs: Sequence[int],
        zplanes: Sequence[int],
) -> Sequence[Tuple[Axes, Sequence[int]]]:
    """
    Given a sequence of axes and their labels, return a sequence of tuples of axes and its
    respective labels.

    For example, if axes_sequence is (ROUND, CH, Z) and each axes has labels [0, 1], return
    ((ROUND, [0, 1]), (CH, [0, 1]), (Z, [0, 1]).
    """
    axes_mapping = {
        Axes.ROUND: rounds,
        Axes.CH: chs,
        Axes.ZPLANE: zplanes,
    }

    return [
        (axes, axes_mapping[axes])
        for axes in axes_order
    ]


def ordered_iterator(
        axes_labels: Sequence[Tuple[Axes, Sequence[int]]]
) -> Iterator[Mapping[Axes, int]]:
    """
    Given a sequence of tuples of axes and its respective sequence of labels, return an iterator
    that steps through all the possible points in the space.  The sequence is ordered from the
    slowest varying axes to the fastest varying axes.
    """
    for tpl in product(*[labels for _, labels in axes_labels]):
        yield dict(zip((index for index, _ in axes_labels), tpl))
