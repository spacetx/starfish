import numpy as np
import pandas as pd
import regional

from starfish.spots import TargetAssignment
from starfish.test.codebook.test_metric_decode import intensity_table_factory
from starfish.types import Features, Indices


def bifurcated_regions(n):
    """
    generate an intensity table with diagonal coordinates ((1, 1), (2, 2), ... (n, n)) over 3
    channels and four rounds, where intensities are randomly generated.

    Split the region into two block-diagonal cells which should encompass 1/2 of the total area but
    all of the points in the domain, since intensities is a diagonal table.
    """

    np.random.seed(777)
    data = np.random.random_sample((n, 3, 4))
    diagonal_intensities = intensity_table_factory(data)

    x = diagonal_intensities[Indices.X.value].max() + 1
    y = diagonal_intensities[Indices.Y.value].max() + 1
    box_one_coords = [
        [0, 0],
        [0, np.floor(x / 2)],
        [np.ceil(y / 2), 0],
        [np.floor(y / 2), np.floor(x / 2)]]
    box_two_coords = [
        [np.floor(y / 2), np.floor(x / 2)],
        [np.floor(y / 2), x],
        [y, np.floor(x / 2)],
        [y, x]
    ]
    regions = regional.many([regional.one(box_one_coords), regional.one(box_two_coords)])

    # assign intensity_table some target values that are just sequential numbers
    diagonal_intensities[Features.TARGET] = (Features.AXIS, np.arange(n).astype(str))

    return diagonal_intensities, regions


def test_target_assignment_point_in_poly_2d_all_points_in_cells():
    """Create a set of intensities where all points lie inside the defined cell regions"""

    diagonal_intensities, regions = bifurcated_regions(10)
    ta = TargetAssignment.PointInPoly2D()

    assigned_intensities = ta.run(diagonal_intensities, regions)

    target_counts = pd.Series(
        *np.unique(assigned_intensities[Features.CELL_ID], return_counts=True)[::-1]
    )

    assert target_counts[0] == 5
    assert target_counts[1] == 5


def test_target_assignment_point_in_poly_2d_points_outside_of_cells():
    """Create a set of intensities where no points fall into cells"""

    n = 10
    diagonal_intensities, regions = bifurcated_regions(n)
    diagonal_intensities[Indices.X.value] += 50  # now all x-values lie outside cell regions
    ta = TargetAssignment.PointInPoly2D()

    assigned_intensities = ta.run(diagonal_intensities, regions)

    assert np.array_equal(
        assigned_intensities[Features.CELL_ID].values,
        np.full(n, fill_value=None, dtype=object)
    )
