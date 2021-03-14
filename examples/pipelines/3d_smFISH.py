"""
.. _smFISH_processing_example:

Processing sequential smFISH in 3D
==================================

This notebook walks through a work flow that analyzes one field of view of a mouse gene panel from
the Allen Institute for Cell Science, using the starfish package.

This example processes an experiment with a single round from a single field of view of sequential
smFISH data taken from mouse primary visual cortex. These data are unpublished, and were kindly
contributed by the Allen Institute for Brain Science as a part of the SpaceTx consortium
project.

The data consist of 45 images from 1 round, 1 channels, and 33 z-planes. Each image is
(2048, 2048) (y, x). There are no test data.

"""

from typing import Optional, Tuple

import starfish
import starfish.data
from starfish import FieldOfView, DecodedIntensityTable
from starfish.types import TraceBuildingStrategies


###################################################################################################
# Define image filters
# --------------------
# The 3d smFISH workflow run by the Allen runs a bandpass filter to remove high and low frequency
# signal and blurs over z with a 1-pixel gaussian to smooth the signal over the z-axis.
#
# low-intensity signal is (stringently) clipped from the images before and after these filters.

# bandpass filter to remove cellular background and camera noise
bandpass = starfish.image.Filter.Bandpass(lshort=.5, llong=7, threshold=0.0)

# gaussian blur to smooth z-axis
glp = starfish.image.Filter.GaussianLowPass(
    sigma=(1, 0, 0),
    is_volume=True
)

# pre-filter clip to remove low-intensity background signal
clip1 = starfish.image.Filter.Clip(p_min=50, p_max=100)

# post-filter clip to eliminate all but the highest-intensity peaks
clip2 = starfish.image.Filter.Clip(p_min=99, p_max=100, is_volume=True)

###################################################################################################
# Define a spot detection method
# ------------------------------
# Spots are detected using a spot finder based on trackpy's locate method, which identifies
# local intensity maxima, and spots are matched to the gene they represent by looking them up in a
# codebook that records which (round, channel) matches which gene target.

tlmpf = starfish.spots.FindSpots.TrackpyLocalMaxPeakFinder(
    spot_diameter=5,  # must be odd integer
    min_mass=0.02,
    max_size=2,  # this is max radius
    separation=7,
    preprocess=False,
    percentile=10,  # this is irrelevant when min_mass, spot_diameter, and max_size are set properly
    verbose=True,
    is_volume=True,
)

###################################################################################################
# Construct the pipeline
# ----------------------

def processing_pipeline(
    experiment: starfish.Experiment,
    fov_name: str,
    n_processes: Optional[int]=None
) -> Tuple[starfish.ImageStack, starfish.IntensityTable]:
    """Process a single field of view of an experiment

    Parameters
    ----------
    experiment : starfish.Experiment
        starfish experiment containing fields of view to analyze
    fov_name : str
        name of the field of view to process
    n_processes : int

    Returns
    -------
    starfish.IntensityTable :
        decoded IntensityTable containing spots matched to the genes they are hybridized against
    """

    all_intensities = list()
    codebook = experiment.codebook

    print("Loading images...")
    images = enumerate(experiment[fov_name].get_images(FieldOfView.PRIMARY_IMAGES))

    decoder = starfish.spots.DecodeSpots.PerRoundMaxChannel(
        codebook=codebook,
        trace_building_strategy=TraceBuildingStrategies.SEQUENTIAL
    )

    for image_number, primary_image in images:
        print(f"Filtering image {image_number}...")
        filter_kwargs = dict(
            in_place=True,
            verbose=True,
            n_processes=n_processes
        )
        clip1.run(primary_image, **filter_kwargs)
        bandpass.run(primary_image, **filter_kwargs)
        glp.run(primary_image, **filter_kwargs)
        clip2.run(primary_image, **filter_kwargs)

        print("Calling spots...")
        spots = tlmpf.run(primary_image)
        print("Decoding spots...")
        decoded_intensities = decoder.run(spots=spots)
        all_intensities.append(decoded_intensities)

    decoded = DecodedIntensityTable.concatenate_intensity_tables(all_intensities)
    decoded = decoded[decoded["total_intensity"] > .025]

    return primary_image, decoded

###################################################################################################
# Load data & run pipeline
# ----------------------------------------

experiment = starfish.data.allen_smFISH(use_test_data=True)

image, intensities = processing_pipeline(experiment, fov_name='fov_001')
