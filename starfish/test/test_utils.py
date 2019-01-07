from collections import OrderedDict

import numpy as np

from starfish import Codebook, ImageStack
from starfish.types import Features, Indices, PhysicalCoordinateTypes


def imagestack_with_coords_factory(stack_shape: OrderedDict, coords: OrderedDict) -> ImageStack:
    """
    Create an ImageStack of given shape and assigns the given x,y,z
    min/max physical coordinates to each tile.

    Parameters
    ----------
    stack_shape: OrderedDict
        Dict[Indices, int] defining the size of each dimension for an ImageStack

    coords: OrderedDict
        Dict[PhysicalCoordinateTypes, float] defining the min/max values of physical
        coordinates to assign to each tile of the return ImageStack
    """

    stack = ImageStack.synthetic_stack(num_round=stack_shape[Indices.ROUND],
                                       num_ch=stack_shape[Indices.CH],
                                       num_z=stack_shape[Indices.Z],
                                       tile_height=stack_shape[Indices.Y],
                                       tile_width=stack_shape[Indices.X])

    coords_array = [coords[PhysicalCoordinateTypes.X_MIN],
                    coords[PhysicalCoordinateTypes.X_MAX],
                    coords[PhysicalCoordinateTypes.Y_MIN],
                    coords[PhysicalCoordinateTypes.Y_MAX],
                    coords[PhysicalCoordinateTypes.Z_MIN],
                    coords[PhysicalCoordinateTypes.Z_MAX]]

    for _round in stack.index_labels(Indices.ROUND):
        for ch in stack.index_labels(Indices.CH):
            for z in stack.index_labels(Indices.Z):
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
