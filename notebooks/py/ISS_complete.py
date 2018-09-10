#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START code
import starfish
# EPY: END code

# EPY: START code
experiment_metadata = '/Users/ajc/Desktop/iss_breast_formatted/experiment.json'
# EPY: END code

# EPY: START code
exp = starfish.Experiment.from_json(experiment_metadata)
# EPY: END code

# EPY: START code
exp
# EPY: END code

# EPY: START code
exp.fov()
# EPY: END code

# EPY: START code
from starfish.image import Filter
from starfish.image import Registration
from starfish.spots import SpotFinder
import warnings
from starfish.image import Segmentation


def pipeline(fov): 
    
    primary_image = fov.primary_image

    # filter raw data
    masking_radius = 15
    filt = Filter.WhiteTophat(masking_radius, is_volume=False)
    print('WhiteTophat Filtering...')
    filt.run(primary_image, verbose=True)
        
    print('Registering...')
    registration = Registration.FourierShiftRegistration(
        upsampling=1000,
        reference_stack=dots
    )
    registration.run(primary_image, verbose=True)
    
    # parameters to define the allowable gaussian sizes (parameter space)
    min_sigma = 1
    max_sigma = 10
    num_sigma = 30
    threshold = 0.01

    p = SpotFinder.GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        measurement_type='mean',
    )

    # detect triggers some numpy warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # blobs = dots; define the spots in the dots image, but then find them again in the stack.
        blobs_image = dots.max_proj(Indices.ROUND, Indices.Z)
        intensities = p.run(primary_image, blobs_image=blobs_image)
        
    decoded = experiment.codebook.decode_per_round_max(intensities)
    
    dapi_thresh = .16  # binary mask for cell (nuclear) locations
    stain_thresh = .22  # binary mask for overall cells // binarization of stain
    min_dist = 57

    stain = np.mean(primary_image.max_proj(Indices.CH, Indices.Z), axis=0)
    stain = stain/stain.max()
    nuclei_projection = nuclei.max_proj(Indices.ROUND, Indices.CH, Indices.Z)

    seg = Segmentation.Watershed(
        dapi_threshold=dapi_thresh,
        input_threshold=stain_thresh,
        min_distance=min_dist
    )
    regions = seg.run(primary_image, nuclei)
    
    return decoded, regions
# EPY: END code
