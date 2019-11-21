from typing import Tuple

import numpy as np
import xarray as xr

from starfish.core.codebook.codebook import Codebook
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.imagestack.test.factories import create_imagestack_from_codebook
from starfish.core.spots.DecodeSpots import MetricDistance, PerRoundMaxChannel, SimpleLookupDecoder
from starfish.core.spots.FindSpots import BlobDetector
from starfish.core.types import Axes, Features


def simple_gaussian_spot_detector() -> BlobDetector:
    """create a basic gaussian spot detector"""
    return BlobDetector(
        min_sigma=1,
        max_sigma=4,
        num_sigma=5,
        threshold=0,
        measurement_type='max')


def composite_codebook() -> Tuple[Codebook, ImageStack]:
    """
    Produce an Imagestack representing a composite experiment where the first 2 rounds are
    multiplexed data and the last round is sequential data.

    Returns
    -------
    Codebook :
        codebook containing codes that match the data
    ImageStack :
        noiseless ImageStack containing one spot per code in codebook
    """
    codebook_data = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 2, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_C"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 2, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_D"
        }
    ]

    codebook = Codebook.from_code_array(codebook_data)
    imagestack = create_imagestack_from_codebook(
        pixel_dimensions=(10, 100, 100),
        spot_coordinates=((4, 10, 90), (5, 90, 10), (6, 90, 10), (7, 90, 10)),
        codebook=codebook
    )
    return codebook, imagestack


def composite_codebook_mixed_round() -> Tuple[Codebook, ImageStack]:
    """
    Produce an Imagestack representing a composite experiment where the first 2 and a half rounds
    are multiplexed data and the last round and a half is sequential data. This represents the type
    of hybrid experiment happening at the allen.

    Returns
    -------
    Codebook :
        codebook containing codes that match the data
    ImageStack :
        noiseless ImageStack containing one spot per code in codebook
    """
    codebook_data = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 2, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_C"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 2, Axes.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_D"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 3, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_E"
        },
    ]

    codebook = Codebook.from_code_array(codebook_data)
    imagestack = create_imagestack_from_codebook(
        pixel_dimensions=(1, 10, 10),
        spot_coordinates=((0, 5, 6), (0, 2, 3), (0, 7, 1), (0, 8, 2), (0, 3, 6)),
        codebook=codebook
    )
    return codebook, imagestack


def compostie_codebook_seperate_stacks() -> Tuple[Codebook, ImageStack, ImageStack]:
    """
    Produce separate different sized ImageStacks containing data from two different experiment
    types (multiplexed and non multiplexed) and one codebook with the information for both.

    Returns
    -------
    Codebook :
        codebook containing codes that match the data
    ImageStack :
        noiseless ImageStack containing one spot per code in codebook
    """

    # 3 round 3 ch multiplexed data
    multiplexed_data = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 2, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    codebook = Codebook.from_code_array(multiplexed_data)
    multiplexed_stack = create_imagestack_from_codebook(
        pixel_dimensions=(10, 100, 100),
        spot_coordinates=((4, 10, 90), (5, 90, 10)),
        codebook=codebook
    )

    # 3 rounds 1 ch non multiplexed data
    sequential_data = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 3, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_C"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 4, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_D"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 5, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_D"
        }
    ]
    codebook = Codebook.from_code_array(sequential_data)
    sequential_stack = create_imagestack_from_codebook(
        pixel_dimensions=(10, 100, 100),
        spot_coordinates=((4, 10, 90), (5, 90, 10), (7, 90, 10)),
        codebook=codebook
    )
    sequential_stack = sequential_stack.sel(indexers={Axes.ROUND: (3, 6)})

    # create codebook with combined target values
    combined = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 2, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 3, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_C"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 4, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_D"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 5, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_E"
        }

    ]
    codebook = Codebook.from_code_array(combined)
    return codebook, multiplexed_stack, sequential_stack


gaussian_spot_detector = simple_gaussian_spot_detector()


def test_composite_codebook_decoding():
    """
    Test a decoding workflow in which one ImageStack is composed of both multiplexed data and
    sequential data in separate rounds. Show that it's easier to select our the two different
    sections and process them separately.
    """
    # create an ImageStack where rounds 0,1 contain multiplexed data and round 2 contains
    # sequential data
    codebook, stack = composite_codebook()

    # select out the different portions of the imagestack
    multiplexed_stack = stack.sel(indexers={Axes.ROUND: (0, 1)})
    sequential_stack = stack.sel(indexers={Axes.ROUND: 2})

    # select out the corresponding portions of the codebooks
    codebook_multiplexed = codebook.get_partial(indexers={Axes.ROUND: (0, 1)})
    codebook_sequential = codebook.get_partial(indexers={Axes.ROUND: 2})

    # find spots
    reference_image = multiplexed_stack.reduce((Axes.ROUND, Axes.CH), func="max")
    multiplexed_spots = gaussian_spot_detector.run(
        image_stack=multiplexed_stack, reference_image=reference_image)

    sequential_spots = gaussian_spot_detector.run(image_stack=sequential_stack)

    # decode
    per_round_decoder = PerRoundMaxChannel(codebook=codebook_multiplexed)
    decoded_multiplexed = per_round_decoder.run(spots=multiplexed_spots)
    assert np.array_equal(decoded_multiplexed[Features.TARGET].values, ['GENE_B', 'GENE_A'])

    simple_lookup_decoder = SimpleLookupDecoder(codebook=codebook_sequential)
    decoded_sequential = simple_lookup_decoder.run(spots=sequential_spots)
    assert np.array_equal(decoded_sequential[Features.TARGET].values, ['GENE_C', 'GENE_D'])


def test_composite_codebook_decoding_seperate_sized_stacks():
    """
    Test a decoding workflow in which two different ImageStacks of differing sizes and experiment
    types are created from the same codebook. Show that it's possible to select out the portion of
    the codebook to use successfully for each ImageStack.
    """

    # create two different sized ImageStacks with the same codebook
    codebook, multiplexed_stack, sequential_stack = compostie_codebook_seperate_stacks()
    codebook_multiplexed = codebook.get_partial(indexers={Axes.ROUND: (0, 2)})

    # find spots
    reference_image = multiplexed_stack.reduce((Axes.ROUND, Axes.CH), func="max")
    multiplexed_spots = gaussian_spot_detector.run(
        image_stack=multiplexed_stack, reference_image=reference_image)

    sequential_spots = gaussian_spot_detector.run(image_stack=sequential_stack)

    metric_decoder = MetricDistance(codebook=codebook_multiplexed,
                                    max_distance=0,
                                    min_intensity=.1,
                                    norm_order=2)

    decoded_multiplexed = metric_decoder.run(spots=multiplexed_spots)

    assert np.array_equal(decoded_multiplexed[Features.TARGET].values, ['GENE_B', 'GENE_A'])

    lookup_decoder = SimpleLookupDecoder(codebook=codebook)
    decoded_sequential = lookup_decoder.run(spots=sequential_spots)

    assert np.array_equal(sorted(decoded_sequential[Features.TARGET].values),
                          ['GENE_C', 'GENE_D', 'GENE_E'])


def test_composite_codebook_mixed_round():
    """
    Test a decoding workflow in which one ImageStack is composed of both multiplexed data and
    sequential data with one round having both. Show that a decoding workflow is possible by
    selecting out three different sections of the stack and processing them separately.
    """
    codebook, stack = composite_codebook_mixed_round()

    # select the multiplexed portion of the stack
    multiplexed_stack = stack.sel(indexers={Axes.ROUND: (0, 2), Axes.CH: (0, 1)})

    # select the the sequential portion of the combined round
    sequential_stack_part_1 = stack.sel(indexers={Axes.ROUND: 2, Axes.CH: (1, 2)})
    # select the rest of the sequential portion of the stack
    sequential_stack_part_2 = stack.sel(indexers={Axes.ROUND: 3, Axes.CH: (0, 2)})

    # select out the multiplexed portion of the codebook
    codebook_multiplexed = codebook.get_partial(indexers={Axes.ROUND: (0, 2), Axes.CH: (0, 1)})

    # find spots
    reference_image = multiplexed_stack.reduce((Axes.ROUND, Axes.CH), func="max")
    multiplexed_spots = gaussian_spot_detector.run(
        image_stack=multiplexed_stack, reference_image=reference_image)

    # the easiest way to process the sequential portion is to process the two parts independently
    sequential_spots_part_1 = gaussian_spot_detector.run(image_stack=sequential_stack_part_1)
    sequential_spots_part_2 = gaussian_spot_detector.run(image_stack=sequential_stack_part_2)

    metric_decoder = MetricDistance(codebook=codebook_multiplexed,
                                    max_distance=0,
                                    min_intensity=.1,
                                    norm_order=2)

    decoded_multiplexed = metric_decoder.run(spots=multiplexed_spots)

    assert np.array_equal(sorted(decoded_multiplexed[Features.TARGET].values),
                          ['GENE_A', 'GENE_B', 'GENE_C'])

    # decode sequential portions independently then combine the results
    lookup_decoder = SimpleLookupDecoder(codebook=codebook)
    decoded_sequential_part_1 = lookup_decoder.run(spots=sequential_spots_part_1)
    decoded_sequential_part_2 = lookup_decoder.run(spots=sequential_spots_part_2)
    decoded_sequential = xr.concat([decoded_sequential_part_1, decoded_sequential_part_2],
                                   dim="features")

    assert np.array_equal(sorted(decoded_sequential[Features.TARGET].values), ['GENE_D', 'GENE_E'])
