from typing import List, Tuple

from starfish.core.types import Axes, Coordinates


AXES = [a.value for a in Axes if a not in (Axes.ROUND, Axes.CH)]
COORDS = [c.value for c in Coordinates]
AXES_ORDER = Axes.ZPLANE, Axes.Y, Axes.X


def _get_axes_names(ndim: int) -> Tuple[List[str], List[str]]:
    """Get needed axes names given the number of dimensions.

    Parameters
    ----------
    ndim : int
        Number of dimensions.

    Returns
    -------
    axes : List[str]
        Axes names.
    coords : List[str]
        Coordinates names.
    """
    if ndim == 2:
        axes = [axis for axis in AXES if axis != Axes.ZPLANE.value]
        coords = [coord for coord in COORDS if coord != Coordinates.Z.value]
    elif ndim == 3:
        axes = AXES
        coords = COORDS
    else:
        raise TypeError('expected 2- or 3-D image')

    return axes, coords
