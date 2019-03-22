#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "starfish", "language": "python", "name": "starfish"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
## Reproduce 3d smFISH results with Starfish
#
#This notebook walks through a work flow that analyzes one field of view of a mouse gene panel from the Allen Institute for Cell Science, using the starfish package.
# EPY: END markdown

# EPY: START markdown
#The 3d smFISH workflow run by the Allen runs a bandpass filter to remove high and low frequency signal and blurs over z with a 1-pixel gaussian to smooth the signal over the z-axis. low-intensity signal is (stringently) clipped from the images before and after these filters.
#
#Spots are then detected using a spot finder based on trackpy's locate method, which identifies local intensity maxima, and spots are matched to the gene they represent by looking them up in a codebook that records which (round, channel) matches which gene target.
# EPY: END markdown

# EPY: START markdown
### Load imports
# EPY: END markdown

# EPY: START code
# EPY: ESCAPE %gui qt5

import os
from typing import Optional, Tuple

# import napari_gui
import numpy as np

import starfish
from starfish import data, FieldOfView, IntensityTable

# EPY: END code

# EPY: START markdown
### Initialize Pipeline Components with pre-selected parameters
# EPY: END markdown

# EPY: START code
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

# peak caller
tlmpf = starfish.spots.SpotFinder.TrackpyLocalMaxPeakFinder(
    spot_diameter=5, # must be odd integer
    min_mass=0.02,
    max_size=2,  # this is max radius
    separation=7,
    noise_size=0.65,  # this is not used because preprocess is False
    preprocess=False,
    percentile=10,  # this is irrelevant when min_mass, spot_diameter, and max_size are set properly
    verbose=True,
    is_volume=True,
)
# EPY: END code

# EPY: START markdown
### Combine pipeline components into a pipeline
# EPY: END markdown

# EPY: START markdown
#Define a function that identifies spots of a field of view.
# EPY: END markdown

# EPY: START code
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

    spot_attributes = IntensityTable.concatanate_intensity_tables(all_intensities)
    print("Decoding spots...")
    decoded = codebook.decode_per_round_max(spot_attributes)
    decoded = decoded[decoded["total_intensity"]>.025]

    return primary_image, decoded
# EPY: END code

# EPY: START markdown
### Run the pipeline on a field of view
# EPY: END markdown

# EPY: START code
experiment = starfish.data.allen_smFISH(use_test_data=True)

image, intensities = processing_pipeline(experiment, fov_name='fov_001')
# EPY: END code

# EPY: START markdown
### Display the results
# EPY: END markdown

# EPY: START code
viewer = starfish.display(image, intensities)
# EPY: END code
