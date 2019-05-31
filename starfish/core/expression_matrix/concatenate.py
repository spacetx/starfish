from typing import Iterable

import xarray as xr

from starfish.core.types import Features
from .expression_matrix import ExpressionMatrix


def concatenate(expression_matrices: Iterable[ExpressionMatrix]) -> ExpressionMatrix:
    """Concatenate IntensityTables produced for different fields of view or across imaging rounds

    Expression Matrices are concatenated along the cells axis, and the resulting arrays are stored
    densely.

    Parameters
    ----------
    expression_matrices : Iterable[ExpressionMatrix]
        iterable (list-like) of expression matrices to combine

    Returns
    -------
    ExpressionMatrix :
        Concatenated expression matrix containing all input cells. Missing gene values are filled
        with np.nan

    See Also
    --------
    Combine_first: http://xarray.pydata.org/en/stable/combining.html#combine

    """
    concatenated_matrix: xr.DataArray = xr.concat(list(expression_matrices), Features.CELLS)
    return ExpressionMatrix(concatenated_matrix)
