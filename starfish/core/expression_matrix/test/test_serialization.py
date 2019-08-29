import random

from starfish.core.codebook.test.factories import codebook_array_factory
from starfish.core.intensity_table.test import factories
from starfish.core.types import Features

NUMBER_SPOTS = 10


def test_save_expression_matrix():

    codebook = codebook_array_factory()

    decoded_intensities = factories.synthetic_decoded_intensity_table(
        codebook,
        num_z=3,
        height=100,
        width=100,
        n_spots=10
    )
    # mock out come cell_ids
    cell_ids = random.sample(range(1, 20), NUMBER_SPOTS)

    decoded_intensities[Features.CELL_ID] = (Features.AXIS, cell_ids)

    expression_matrix = decoded_intensities.to_expression_matrix()

    # test all saving methods
    expression_matrix.save("expression")
