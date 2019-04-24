"""
ISS Processing Workflow
=======================

"""
import os

import starfish
from starfish.image import ApplyTransform, Filter, LearnTransform, Segment
from starfish.spots import DetectSpots, AssignTargets
from starfish.types import Axes

test = os.getenv("TESTING") is not None


def iss_pipeline(fov, codebook):
    primary_image = fov.get_image(starfish.FieldOfView.PRIMARY_IMAGES)

    dots = primary_image.max_proj(Axes.CH)
    max_dots = dots.max_proj(Axes.ROUND)
    # register the raw image
    learn_translation = LearnTransform.Translation(reference_stack=max_dots,
                                                   axes=Axes.ROUND, upsampling=100)
    transforms_list = learn_translation.run(primary_image.max_proj(Axes.CH, Axes.ZPLANE))
    warp = ApplyTransform.Warp()
    registered = warp.run(primary_image, transforms_list=transforms_list,  in_place=False, verbose=True)

    # filter raw data
    masking_radius = 15
    filt = Filter.WhiteTophat(masking_radius, is_volume=False)
    filtered = filt.run(registered, verbose=True, in_place=False)

    # detect spots using laplacian of gaussians approach
    p = DetectSpots.BlobDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=0.01,
        measurement_type='mean',
    )

    intensities = p.run(
        filtered,
        blobs_image=dots,
        blobs_axes=(Axes.ROUND, Axes.ZPLANE))

    # decode the pixel traces using the codebook
    decoded = codebook.decode_per_round_max(intensities)

    # segment cells
    seg = Segment.Watershed(
        nuclei_threshold=.16,
        input_threshold=.22,
        min_distance=57,
    )
    label_image = seg.run(primary_image, fov.get_image('nuclei'))

    # assign spots to cells
    ta = AssignTargets.Label()
    assigned = ta.run(label_image, decoded)

    return assigned, label_image


# process all the fields of view, not just one
def process_experiment(experiment: starfish.Experiment):
    decoded_intensities = {}
    regions = {}
    for i, (name_, fov) in enumerate(experiment.items()):
        print("processing fov: " + name_)
        decoded, segmentation_results = iss_pipeline(fov, experiment.codebook)
        decoded_intensities[name_] = decoded
        regions[name_] = segmentation_results
    return decoded_intensities, regions


# run the script
exp = starfish.Experiment.from_json("https://d2nhj9g34unfro.cloudfront.net/xiaoyan_qian/ISS_human_HCA_07_MultiFOV/main_files/experiment.json")
decoded_intensities, regions = process_experiment(exp)
