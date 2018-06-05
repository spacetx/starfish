from typing import Any, Callable
from copy import deepcopy

import numpy as np
import pytest
from slicedimage import Tile, TileSet

from starfish.constants import Indices, Coordinates
from starfish.image import ImageStack
from starfish.io import Stack
from starfish.util.synthesize import synthesize


# TODO ambrosejcarr: all fixtures should emit a stack and a codebook
@pytest.fixture(scope='session')
def merfish_stack() -> Stack:
    """retrieve MERFISH testing data from cloudfront and expose it at the module level

    Notes
    -----
    Because download takes time, this fixture runs once per session -- that is, the download is run only once.

    Returns
    -------
    Stack :
        starfish.io.Stack object containing MERFISH data
    """
    s = Stack()
    s.read('https://s3.amazonaws.com/czi.starfish.data.public/20180607/test/MERFISH/fov_001/experiment_new.json')
    return deepcopy(s)


def default_tile_data_provider(hyb: int, ch: int, z: int, height: int, width: int) -> np.ndarray:
    """
    Returns a tile of just ones for any given hyb/ch/z.
    """
    return np.ones((height, width))


def default_tile_extras_provider(hyb: int, ch: int, z: int) -> Any:
    """
    Returns None for extras for any given hyb/ch/z.
    """
    return None


DEFAULT_NUM_HYB = 2
DEFAULT_NUM_CH = 3
DEFAULT_NUM_Z = 4
DEFAULT_HEIGHT = 30
DEFAULT_WIDTH = 20


def synthetic_stack(
        num_hyb: int=DEFAULT_NUM_HYB,
        num_ch: int=DEFAULT_NUM_CH,
        num_z: int=DEFAULT_NUM_Z,
        tile_height: int=DEFAULT_HEIGHT,
        tile_width: int=DEFAULT_WIDTH,
        tile_data_provider: Callable[[int, int, int, int, int], np.ndarray]=default_tile_data_provider,
        tile_extras_provider: Callable[[int, int, int], Any]=default_tile_extras_provider
) -> ImageStack:
    """generate a synthetic ImageStack

    Returns
    -------
    ImageStack :
        imagestack containing a tensor of (2, 3, 4, 30, 20) whose values are all 1.

    """
    img = TileSet(
        {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
        {
            Indices.HYB: num_hyb,
            Indices.CH: num_ch,
            Indices.Z: num_z,
        },
        default_tile_shape=(tile_height, tile_width),
    )
    for hyb in range(num_hyb):
        for ch in range(num_ch):
            for z in range(num_z):
                tile = Tile(
                    {
                        Coordinates.X: (0.0, 0.001),
                        Coordinates.Y: (0.0, 0.001),
                        Coordinates.Z: (0.0, 0.001),
                    },
                    {
                        Indices.HYB: hyb,
                        Indices.CH: ch,
                        Indices.Z: z,
                    },
                    extras=tile_extras_provider(hyb, ch, z),
                )
                tile.numpy_array = tile_data_provider(hyb, ch, z, tile_height, tile_width)

                img.add_tile(tile)

    stack = ImageStack(img)
    return stack


def labeled_synthetic_dataset() -> Stack:
    stack, codebook = synthesize()
    return stack
