from collections import OrderedDict
from typing import Sequence, Tuple

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from starfish.codebook.codebook import Codebook
from starfish.experiment.builder import build_image, TileFetcher
from starfish.experiment.builder.defaultproviders import OnesTile, tile_fetcher_factory
from starfish.imagestack import physical_coordinate_calculator
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes, Coordinates, Features, PhysicalCoordinateTypes


def synthetic_stack(
        num_round: int = 4,
        num_ch: int = 4,
        num_z: int = 12,
        tile_height: int = 50,
        tile_width: int = 40,
        tile_fetcher: TileFetcher = None,
) -> ImageStack:
    """generate a synthetic ImageStack

    Returns
    -------
    ImageStack :
        imagestack containing a tensor whose default shape is (2, 3, 4, 30, 20)
        and whose default values are all 1.

    """
    if tile_fetcher is None:
        tile_fetcher = tile_fetcher_factory(
            OnesTile,
            False,
            {Axes.Y: tile_height, Axes.X: tile_width},
        )

    collection = build_image(
        range(1),
        range(num_round),
        range(num_ch),
        range(num_z),
        tile_fetcher,
    )
    tileset = list(collection.all_tilesets())[0][1]

    return ImageStack.from_tileset(tileset)


def imagestack_with_coords_factory(stack_shape: OrderedDict, coords: OrderedDict) -> ImageStack:
    """
    Create an ImageStack of given shape and assigns the given x,y,z
    min/max physical coordinates to each tile.

    Parameters
    ----------
    stack_shape: OrderedDict
        Dict[Axes, int] defining the size of each dimension for an ImageStack

    coords: OrderedDict
        Dict[PhysicalCoordinateTypes, float] defining the min/max values of physical
        coordinates to assign to the Imagestack
    """

    stack = synthetic_stack(
        num_round=stack_shape[Axes.ROUND],
        num_ch=stack_shape[Axes.CH],
        num_z=stack_shape[Axes.ZPLANE],
        tile_height=stack_shape[Axes.Y],
        tile_width=stack_shape[Axes.X],
    )

    stack.xarray[Coordinates.X.value] = xr.DataArray(
        np.linspace(coords[PhysicalCoordinateTypes.X_MIN], coords[PhysicalCoordinateTypes.X_MAX],
                    stack.xarray.sizes[Axes.X.value]), dims=Axes.X.value)

    stack.xarray[Coordinates.Y.value] = xr.DataArray(
        np.linspace(coords[PhysicalCoordinateTypes.Y_MIN], coords[PhysicalCoordinateTypes.Y_MAX],
                    stack.xarray.sizes[Axes.Y.value]), dims=Axes.Y.value)

    z_coord = physical_coordinate_calculator.\
        get_physical_coordinates_of_z_plane((coords[PhysicalCoordinateTypes.Z_MIN],
                                             coords[PhysicalCoordinateTypes.Z_MAX]))

    stack.xarray[Coordinates.Z.value] = xr.DataArray(np.zeros(
        stack.xarray.sizes[Axes.ZPLANE.value]),
        dims=Axes.ZPLANE.value)

    for z in stack.axis_labels(Axes.ZPLANE):
        stack.xarray[Coordinates.Z.value].loc[z] = z_coord

    return stack


def create_imagestack_from_codebook(
    pixel_dimensions: Tuple[int, int, int],
    spot_coordinates: Sequence[Tuple[int, int, int]],
    codebook: Codebook
) -> ImageStack:
    """
    creates a numpy array containing one spot per codebook entry at spot_coordinates. length of
    spot_coordinates must therefore match the number of codes in Codebook.
    """
    assert len(spot_coordinates) == codebook.sizes[Features.TARGET]

    data_shape = (
        codebook.sizes[Axes.ROUND.value],
        codebook.sizes[Axes.CH.value],
        *pixel_dimensions
    )
    imagestack_data = np.zeros(data_shape, dtype=np.float32)

    for ((z, y, x), f) in zip(spot_coordinates, range(codebook.sizes[Features.TARGET])):
        imagestack_data[:, :, z, y, x] = codebook[f].transpose(Axes.ROUND.value, Axes.CH.value)

    # blur with a small non-isotropic kernel TODO make kernel smaller.
    imagestack_data = gaussian_filter(imagestack_data, sigma=(0, 0, 0.7, 1.5, 1.5))
    return ImageStack.from_numpy_array(imagestack_data)
