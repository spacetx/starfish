from collections import OrderedDict

import numpy as np

from starfish import Codebook, ImageStack
from starfish.types import Features, Indices


def imagestack_with_coords_factory(stack_shape: OrderedDict, coords: OrderedDict) -> ImageStack:
    """
    Create an ImageStack of given shape and assigns the given x,y,z
    min/max physical coordinates to each tile.

    Parameters
    ----------
    stack_shape: OrderedDict
        Dict[Indices, int] defining hte size of each dimension for an ImageStack

    coords: OrderedDict
        Dict[PhysicalCoordinateTypes, float] defining the mon/max values of physical
        coordinates to assign to each tile of the return ImageStack
    """

    tuple_shape = tuple(t[1] for t in stack_shape.items())

    stack = ImageStack.synthetic_stack(*tuple_shape)

    coords_array = list(t[1] for t in coords.items())

    for _round in range(stack.num_rounds):
        for ch in range(stack.num_chs):
            for z in range(stack.num_zlayers):
                coordinate_selector = {
                    Indices.ROUND.value: _round,
                    Indices.CH.value: ch,
                    Indices.Z.value: z,
                }

                stack._coordinates.loc[coordinate_selector] = np.array(coords_array)

    return stack


def codebook_array_factory() -> Codebook:
    """
    Codebook with two codewords describing an experiment with three channels and two imaging rounds.
    Both codes have two "on" channels.
    """
    data = [
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 0, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 2, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    return Codebook.from_code_array(data)
