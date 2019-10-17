import numpy as np

from starfish.core.spots.DecodeSpots import trace_builders
from starfish.core.spots.FindSpots import BlobDetector
from starfish.core.test.factories import composite_codebook
from starfish.core.types import Axes, Features


def simple_gaussian_spot_detector() -> BlobDetector:
    """create a basic gaussian spot detector"""
    return BlobDetector(
        min_sigma=1,
        max_sigma=4,
        num_sigma=5,
        threshold=0,
        measurement_type='max')


gaussian_spot_detector = simple_gaussian_spot_detector()


def test_composite_codebook_decoding():
    # create an Imagestack where rounds 0,1 are multiplexed and round 2 is sequential
    codebook, stack = composite_codebook()

    multiplexed_stack = stack.sel(indexers={Axes.ROUND: (0, 1)})
    sequential_stack = stack.sel(indexers={Axes.ROUND: 2})

    codebook_multiplexed = codebook.get_partial(indexers={Axes.ROUND: (0, 1)})
    codebook_sequential = codebook.get_partial(indexers={Axes.ROUND: 2})

    reference_image = multiplexed_stack.reduce((Axes.ROUND, Axes.CH), func="max")
    multiplexed_spots = gaussian_spot_detector.run(
        image_stack=multiplexed_stack, reference_image=reference_image)

    sequential_spots = gaussian_spot_detector.run(image_stack=sequential_stack)

    # so for rounds 0 and 1 we wanna build traces exact match but for round 2 we wanna build
    # them sequentially
    multiplexed_traces = trace_builders.build_spot_traces_exact_match(multiplexed_spots)

    sequential_traces = trace_builders.build_traces_sequential(sequential_spots)

    decoded_multiplexed = codebook_multiplexed.decode_per_round_max(multiplexed_traces)
    assert np.array_equal(decoded_multiplexed[Features.TARGET].values, ['GENE_B', 'GENE_A'])

    decoded_sequential = codebook_sequential.decode_per_round_max(sequential_traces)
    assert np.array_equal(decoded_sequential[Features.TARGET].values, ['GENE_C', 'GENE_D'])
