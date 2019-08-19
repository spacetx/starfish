import random

from starfish import DecodedIntensityTable
from starfish.core.codebook.test.factories import codebook_array_factory
from starfish.core.types import Features

NUMBER_SPOTS = 10


def test_save_expression_matrix():

    codebook = codebook_array_factory()

    decoded_intensities = DecodedIntensityTable.synthetic_intensities(
        codebook,
        num_z=3,
        height=100,
        width=100,
        n_spots=10
    )
    # mock out come cell_ids
    cell_ids = random.sample(range(1, 20), NUMBER_SPOTS)

    decoded_intensities.assign_cell_ids(cell_ids=(Features.AXIS, cell_ids))

    expression_matrix = decoded_intensities.to_expression_matrix()

    # test all saving methods
    expression_matrix.save("expression")
