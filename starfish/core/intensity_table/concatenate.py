from typing import Iterable

import xarray as xr

from starfish.core.types import Features
from .intensity_table import IntensityTable


def concatenate(intensity_tables: Iterable[IntensityTable]) -> IntensityTable:
    """Concatenate IntensityTables produced for different fields of view or across imaging rounds

    IntensityTables are concatenated along the features axis, and the resulting arrays are stored
    densely, even if the underlying data is sparse, since xarray does not yet support sparse array
    structures. This means that spots that are identified in different rounds and channels will
    be identified as separate features, even if they have exactly identical coordinates.

    To merge spots that share coordinates across rounds and channels into single features amenable
    to decoding, use IntensityTable.combine_first()

    Parameters
    ----------
    intensity_tables : Iterable[IntensityTable]
        iterable (list-like) of intensity tables to combine

    Returns
    -------
        merged IntensityTable. Missing values are filled with np.NaN

    See Also
    --------
    Sparse Arrays in xarray: https://github.com/pydata/xarray/issues/1375
    Combine_first: http://xarray.pydata.org/en/stable/combining.html#combine
    """
    concatenated_intensities: xr.DataArray = xr.concat(list(intensity_tables), Features.AXIS)
    return IntensityTable(concatenated_intensities)
