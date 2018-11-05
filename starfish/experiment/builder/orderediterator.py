from itertools import product
from typing import Iterator, Mapping, Sequence, Tuple

from starfish.types import Indices


def join_dimension_sizes(
        dimension_order: Sequence[Indices],
        *,
        size_for_round: int,
        size_for_ch: int,
        size_for_z: int,
) -> Sequence[Tuple[Indices, int]]:
    """
    Given a sequence of dimensions and their sizes, return a sequence of tuples of dimensions and
    its respective size.

    For example, if dimension_sequence is (ROUND, CH, Z) and each dimension is of size 2, return
    ((ROUND, 2), (CH, 2), (Z, 2)).
    """
    dimension_mapping = {
        Indices.ROUND: size_for_round,
        Indices.CH: size_for_ch,
        Indices.Z: size_for_z,
    }

    return [
        (dimension, dimension_mapping[dimension])
        for dimension in dimension_order
    ]


def ordered_iterator(
        dimension_sizes: Sequence[Tuple[Indices, int]]
) -> Iterator[Mapping[Indices, int]]:
    """
    Given a sequence of tuples of dimensions and its respective size, return an iterator that steps
    through all the possible points in the space.  The sequence is ordered from the slowest varying
    dimension to the fastest varying dimension.
    """
    for tpl in product(*[range(dimension_sizes) for _, dimension_sizes in dimension_sizes]):
        yield dict(zip((index for index, _ in dimension_sizes), tpl))
