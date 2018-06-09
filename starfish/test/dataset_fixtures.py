from typing import Any, Callable

import numpy as np
import pytest
from slicedimage import Tile, TileSet

from starfish.constants import Indices, Coordinates
from starfish.image import ImageStack
from starfish.io import Stack


@pytest.fixture(scope='session')
def merfish_stack() -> Stack:
    """retrieve MERFISH testing data from cloudfront and expose it at the module level

    Notes
    -----
    Because download takes time, this fixture runs once per session -- that is, the download is run only once.
    Therefore, methods consuming this fixture should COPY the data using deepcopy before executing code that changes
    the data, as otherwise this can affect other tests and cause failure cascades.

    Returns
    -------
    Stack :
        starfish.io.Stack object containing MERFISH data
    """
    s = Stack()
    s.read('https://s3.amazonaws.com/czi.starfish.data.public/20180607/test/MERFISH/fov_001/experiment_new.json')
    return s


@pytest.fixture(scope='session')
def synthetic_stack_factory():
    """
    Inject this factory, which is a method to produce image stacks.
    """
    def synthetic_stack() -> ImageStack:
        """generate a synthetic ImageStack

        Returns
        -------
        ImageStack :
            imagestack containing a tensor of (2, 3, 4, 30, 20) whose values are all 1.

        """
        NUM_HYB = 2
        NUM_CH = 3
        NUM_Z = 4
        Y = 30
        X = 20

        img = TileSet(
            {Coordinates.X, Coordinates.Y, Indices.HYB, Indices.CH, Indices.Z},
            {
                Indices.HYB: NUM_HYB,
                Indices.CH: NUM_CH,
                Indices.Z: NUM_Z,
            },
            default_tile_shape=(Y, X),
        )
        for hyb in range(NUM_HYB):
            for ch in range(NUM_CH):
                for z in range(NUM_Z):
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
                    )
                    tile.numpy_array = np.ones((Y, X))

                    img.add_tile(tile)

        stack = ImageStack(img)
        return stack

    return synthetic_stack
