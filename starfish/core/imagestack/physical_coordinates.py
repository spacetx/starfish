from typing import Tuple


def _get_physical_coordinates_of_z_plane(zrange: Tuple[float, float]):
    """Calculate the midpoint of the given zrange."""
    physical_z = (zrange[1] - zrange[0]) / 2 + zrange[0]
    return physical_z
