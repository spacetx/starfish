from collections import OrderedDict

import numpy as np

from starfish import Codebook, ImageStack
from starfish.types import Axes, Features, PhysicalCoordinateTypes


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
        coordinates to assign to each tile of the return ImageStack
    """

    stack = ImageStack.synthetic_stack(num_round=stack_shape[Axes.ROUND],
                                       num_ch=stack_shape[Axes.CH],
                                       num_z=stack_shape[Axes.ZPLANE],
                                       tile_height=stack_shape[Axes.Y],
                                       tile_width=stack_shape[Axes.X])

    coords_array = [coords[PhysicalCoordinateTypes.X_MIN],
                    coords[PhysicalCoordinateTypes.X_MAX],
                    coords[PhysicalCoordinateTypes.Y_MIN],
                    coords[PhysicalCoordinateTypes.Y_MAX],
                    coords[PhysicalCoordinateTypes.Z_MIN],
                    coords[PhysicalCoordinateTypes.Z_MAX]]

    for _round in stack.axis_labels(Axes.ROUND):
        for ch in stack.axis_labels(Axes.CH):
            for z in stack.axis_labels(Axes.ZPLANE):
                coordinate_selector = {
                    Axes.ROUND.value: _round,
                    Axes.CH.value: ch,
                    Axes.ZPLANE.value: z,
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
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 2, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    return Codebook.from_code_array(data)
