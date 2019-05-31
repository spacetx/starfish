import numpy as np

from starfish.core.expression_matrix.concatenate import concatenate
from starfish.core.expression_matrix.expression_matrix import ExpressionMatrix
from starfish.types import Features


def test_concatenate_two_expression_matrices():
    a_data = np.array(
        [[0, 1],
         [1, 0]]
    )
    b_data = np.array(
        [[0],
         [1]]
    )
    dims = [Features.CELLS, Features.GENES]
    a_coords = [(Features.CELLS, [0, 1]), (Features.GENES, ["x", "y"])]
    b_coords = [(Features.CELLS, [0, 1]), (Features.GENES, ["x"])]

    a = ExpressionMatrix(a_data, dims=dims, coords=a_coords)
    b = ExpressionMatrix(b_data, dims=dims, coords=b_coords)

    concatenated = concatenate([a, b])

    expected = np.array(
        [[0, 1],
         [1, 0],
         [0, np.nan],
         [1, np.nan]]
    )

    np.testing.assert_equal(concatenated.values, expected)
