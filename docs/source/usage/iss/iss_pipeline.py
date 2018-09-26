import os

import starfish
from starfish import data
from starfish.image import Filter, Registration, Segmentation
from starfish.spots import SpotFinder, TargetAssignment
from starfish.types import Indices


def iss_pipeline(fov, codebook):
    primary_image = fov.primary_image

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
    p = SpotFinder.GaussianSpotDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=0.01,
        measurement_type='mean',
    )
    blobs_image = fov['dots'].max_proj(Indices.ROUND, Indices.Z)
    intensities = p.run(filtered, blobs_image=blobs_image)

    # decode the pixel traces using the codebook
    decoded = codebook.decode_per_round_max(intensities)

    # segment cells
    seg = Segmentation.Watershed(
        dapi_threshold=.16,
        input_threshold=.22,
        min_distance=57,
    )
    regions = seg.run(primary_image, fov['nuclei'])

    # assign spots to cells
    ta = TargetAssignment.PointInPoly2D()
    assigned = ta.run(decoded, regions)

    return assigned, regions


# process all the fields of view, not just one
def process_experiment(experiment: starfish.Experiment):
    decoded_intensities = {}
    regions = {}
    for name_, fov in experiment.items():
        decoded, segmentation_results = iss_pipeline(fov, experiment.codebook)
        decoded_intensities[name_] = decoded
        regions[name_] = segmentation_results
    return decoded_intensities, regions


# run the script
# test = os.getenv("USE_TEST_DATA") is not None
# if test:
#     exp = data.ISS(test_data=True)
# else:
exp = starfish.Experiment.from_json("iss/formatted/experiment.json")
decoded_intensities, regions = process_experiment(exp)
