from typing import List, Tuple

from starfish.core.types import Axes, Coordinates


def _get_axes_names(ndim: int) -> Tuple[List[Axes], List[Coordinates]]:
    """Get needed axes and coordinates given the number of dimensions.

    Parameters
    ----------
    ndim : int
        Number of dimensions.

    Returns
    -------
    axes : List[Axes]
        Axes.
    coords : List[Coordinates]
        Coordinates.
    """
    if ndim == 2:
        axes = [Axes.Y, Axes.X]
        coords = [Coordinates.Y, Coordinates.X]
    elif ndim == 3:
        axes = [Axes.ZPLANE, Axes.Y, Axes.X]
        coords = [Coordinates.Z, Coordinates.Y, Coordinates.X]
    else:
        raise TypeError('expected 2- or 3-D image')

    return axes, coords
