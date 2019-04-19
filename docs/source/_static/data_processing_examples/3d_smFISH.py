"""
Single Field of View for sequential single-molecule FISH processed in 3d
========================================================================

This notebook walks through a work flow that analyzes one field of view of a mouse gene panel from
the Allen Institute for Cell Science, using the starfish package.
"""

from typing import Optional, Tuple
from IPython import get_ipython

import starfish
import starfish.data
from starfish import FieldOfView, IntensityTable

# equivalent to %gui qt
ipython = get_ipython()
ipython.magic("gui qt5")


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

tlmpf = starfish.spots.DetectSpots.TrackpyLocalMaxPeakFinder(
    spot_diameter=5,  # must be odd integer
    min_mass=0.02,
    max_size=2,  # this is max radius
    separation=7,
    noise_size=0.65,  # this is not used because preprocess is False
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

    print("Loading images...")
    primary_image = experiment[fov_name].get_image(FieldOfView.PRIMARY_IMAGES)
    all_intensities = list()
    codebook = experiment.codebook

    for primary_image in experiment[fov_name].iterate_image_type(FieldOfView.PRIMARY_IMAGES):

        print("Filtering images...")
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
        spot_attributes = tlmpf.run(primary_image)
        all_intensities.append(spot_attributes)

    spot_attributes = IntensityTable.concatenate_intensity_tables(all_intensities)

    print("Decoding spots...")
    decoded = codebook.decode_per_round_max(spot_attributes)
    decoded = decoded[decoded["total_intensity"] > .025]

    return primary_image, decoded

###################################################################################################
# Load data, run pipeline, display results
# ----------------------------------------

experiment = starfish.data.allen_smFISH(use_test_data=True)

image, intensities = processing_pipeline(experiment, fov_name='fov_001')

# uncomment the below line to visualize the output with the spot calls.
# viewer = starfish.display(image, intensities)
