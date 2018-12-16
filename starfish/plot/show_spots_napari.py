from numpy import array, zeros

from starfish import ImageStack, IntensityTable
from starfish.types import Indices


def show_spots_napari(
    spots: IntensityTable, background_image: ImageStack, radius_multiplier: int=30
) -> None:
    """Display detected spots on a background image using Napari

    Parameters
    ----------
    spots : IntensityTable
        IntensityTable containing spot information. Will be projected to match the coordinates of
        the background image, if provided.
    background_image : ImageStack
        ImageStack containing data to display spots on top of.
    radius_multiplier : int
        Multiplies the radius of the displayed spots (default 30)
    """

    mp = background_image.max_proj(Indices.CH, Indices.ROUND)

    # Make the array of maker coordinates
    # If the background image is z projected, also z project the coordinates
    c_r = zeros(len(spots.y.values))
    if mp.raw_shape[2] == 1:
        coords = array([spots.x.values, spots.y.values, c_r, c_r, c_r]).T

    else:
        coords = array([spots.x.values, spots.y.values, c_r, c_r, spots.z.values]).T

    # Create the Napari viewer with an image stack
    viewer, axes = mp.show_stack_napari({})

    # This initializes an index for the instantiated marker display
    # Should be fixed in Napari in the future - KY
    viewer._index = [0, 0, 0, 0, 0]

    # Get the sizes
    sizes = spots.radius.values * radius_multiplier

    # Add the markers to the viewer
    viewer.add_markers(
        coords=coords, face_color='white', edge_color='white', symbol='ring', size=sizes
    )

    return viewer
