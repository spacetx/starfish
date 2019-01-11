import os

import starfish
from starfish.image import Filter, Registration, Segmentation
from starfish.spots import SpotFinder, TargetAssignment
from starfish.types import Axes

test = os.getenv("TESTING") is not None


def iss_pipeline(fov, codebook):
    primary_image = fov[starfish.FieldOfView.PRIMARY_IMAGES]

    # register the raw images
    registration = Registration.FourierShiftRegistration(
        upsampling=1000,
        reference_stack=fov['dots']
    )
    registered = registration.run(primary_image, in_place=False)

    # filter raw data
    masking_radius = 15
    filt = Filter.WhiteTophat(masking_radius, is_volume=False)
    filtered = filt.run(registered, verbose=True, in_place=False)

    # detect spots using laplacian of gaussians approach
    p = SpotFinder.BlobDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=0.01,
        measurement_type='mean',
    )

    mp = fov['dots'].max_proj(Axes.ROUND, Axes.ZPLANE)
    mp_numpy = mp._squeezed_numpy(Axes.ROUND, Axes.ZPLANE)
    intensities = p.run(filtered, blobs_image=mp_numpy)

    # decode the pixel traces using the codebook
    decoded = codebook.decode_per_round_max(intensities)

    # segment cells
    seg = Segmentation.Watershed(
        nuclei_threshold=.16,
        input_threshold=.22,
        min_distance=57,
    )
    label_image = seg.run(primary_image, fov['nuclei'])

    # assign spots to cells
    ta = TargetAssignment.Label()
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
    # TODO: (ttung) Pending a fix for https://github.com/spacetx/starfish/issues/700, it's not
    # possible to validate the schema for this experiment.
    with starfish.config.environ(VALIDATION_STRICT="false"):
        exp = starfish.Experiment.from_json(
            "https://d2nhj9g34unfro.cloudfront.net/browse/formatted/20180926/iss_breast/experiment.json")
else:
    exp = starfish.Experiment.from_json("iss/formatted/experiment.json")
decoded_intensities, regions = process_experiment(exp)
