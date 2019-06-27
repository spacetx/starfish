import random

from starfish import IntensityTable
from starfish.core.codebook.test.factories import codebook_array_factory
from starfish.core.types import Features

NUMBER_SPOTS = 10


def test_save_expression_matrix():

    codebook = codebook_array_factory()

    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=3,
        height=100,
        width=100,
        n_spots=10
    )
    # mock out come cell_ids
    cell_ids = random.sample(range(1, 20), NUMBER_SPOTS)
    intensities[Features.CELL_ID] = (Features.AXIS, cell_ids)

    expression_matrix = intensities.to_expression_matrix()

    # test all saving methods
    expression_matrix.save("expression")
