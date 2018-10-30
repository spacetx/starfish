import numpy as np

from starfish import Codebook, ImageStack, IntensityTable
from starfish.imagestack import physical_coordinate_calculator
from starfish.types import Features, Indices


def imagestack_with_coords_factory(stack_shape, coords) -> ImageStack:
    """
    Create an ImageStack and sets the same coords on every tile
    """
    stack = ImageStack.synthetic_stack(*stack_shape)

    for _round in range(stack.num_rounds):
        for ch in range(stack.num_chs):
            for z in range(stack.num_zlayers):
                coordinate_selector = {
                    Indices.ROUND.value: _round,
                    Indices.CH.value: ch,
                    Indices.Z.value: z,
                }

                stack._coordinates.loc[coordinate_selector] = np.array(coords)

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


def test_tranfering_physical_coords_to_intensity_table():
    stack = imagestack_with_coords_factory((3, 2, 1, 50, 40), [1, 2, 4, 6, 1, 3])
    codebook = codebook_array_factory()

    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=1,
        height=50,
        width=40,
        n_spots=2
    )

    intensities = physical_coordinate_calculator.\
        transfer_physical_coords_from_imagestack_to_intensity_table(stack, intensities)
