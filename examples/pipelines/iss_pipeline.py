"""
.. _iss_processing_example:

ISS Processing Workflow
=======================

"""
import os

import starfish
from starfish.image import ApplyTransform, Filter, LearnTransform, Segment
from starfish.spots import FindSpots, DecodeSpots, AssignTargets
from starfish.types import Axes, FunctionSource

test = os.getenv("TESTING") is not None


def iss_pipeline(fov, codebook):
    primary_image = fov.get_image(starfish.FieldOfView.PRIMARY_IMAGES)

    # register the raw image
    learn_translation = LearnTransform.Translation(reference_stack=fov.get_image('dots'),
                                                   axes=Axes.ROUND, upsampling=100)
    transforms_list = learn_translation.run(
        primary_image.reduce({Axes.CH, Axes.ZPLANE}, func="max"))
    warp = ApplyTransform.Warp()
    registered = warp.run(primary_image, transforms_list=transforms_list,  in_place=False, verbose=True)

    # filter raw data
    masking_radius = 15
    filt = Filter.WhiteTophat(masking_radius, is_volume=False)
    filtered = filt.run(registered, verbose=True, in_place=False)

    bd = FindSpots.BlobDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=0.01,
        measurement_type='mean',
    )

    # detect spots using laplacian of gaussians approach
    dots_max = fov.get_image('dots').reduce((Axes.ROUND, Axes.ZPLANE), func="max")
    # locate spots in a reference image
    spots = bd.run(reference_image=dots_max, image_stack=filtered)

    # decode the pixel traces using the codebook
    decoder = DecodeSpots.PerRoundMaxChannel(codebook=codebook)
    decoded = decoder.run(spots=spots)

    # segment cells
    seg = Segment.Watershed(
        nuclei_threshold=.16,
        input_threshold=.22,
        min_distance=57,
    )
    label_image = seg.run(primary_image, fov.get_image('dots'))

    # assign spots to cells
    ta = AssignTargets.Label()
    assigned = ta.run(label_image, decoded)

    return assigned, label_image


# process all the fields of view, not just one
def process_experiment(experiment: starfish.Experiment):
    decoded_intensities = {}
    regions = {}
    for i, (name_, fov) in enumerate(experiment.items()):
        decoded, segmentation_results = iss_pipeline(fov, experiment.codebook)
        decoded_intensities[name_] = decoded
        regions[name_] = segmentation_results
        if test and i == 1:
            # only run through 2 fovs for the test
            break
    return decoded_intensities, regions


# run the script
if test:
    exp = starfish.Experiment.from_json(
        "https://d2nhj9g34unfro.cloudfront.net/browse/formatted/20180926/iss_breast/experiment.json")
else:
    exp = starfish.Experiment.from_json("iss/formatted/experiment.json")
decoded_intensities, regions = process_experiment(exp)
