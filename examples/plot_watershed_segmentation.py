"""
.. _tutorial_watershed_segmentation:

Watershed Segmentation
======================

In order to create a cell by gene expression matrix from image-based transcriptomics data, RNA
spots must be assigned to cells. One approach is to binarize images of stained cellular structures
to define cell boundaries. A problem arises when there are contiguous or partially overlapping
cells, as is often the case in tissue. These clumped cells are mislabeled as one cell by connected
component labeling.

Watershed segmentation can be used to divide connected objects like clumped cells by finding
watershed lines that separate pixel intensity basins. The classic method for computing pixel
intensity values from a binary image is applying a distance transform, which labels foreground
pixels furthest from the background with the lowest values and pixels close to the background
with higher values. The overall effect after watershed is to segment objects into convex shapes,
which is generally a good approximation for a cell or nucleus. More details about the watershed
algorithm can be found
`here <https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html>`_.

Starfish allows users to implement watershed segmentation into their pipelines in two ways. The
first is to use :py:class:`starfish.morphology.Segment.WatershedSegment` to explicitly define a
segmentation pipeline. The second is to use :py:class:`starfish.image.Segment.Watershed`,
a pre-built watershed segmentation pipeline that uses watershed segmented nuclei as seeds to run
watershed segmentation on cell stain images.

The pre-built watershed segmentation pipeline will not work for everyone, so this tutorial
will first show you how you can define a custom watershed segmentation pipeline. Then this tutorial
will cover how to run the pre-built segmentation algorithm.

Input requirements:
* registered primary images :py:class:`ImageStack`
* registered nuclei images :py:class:`ImageStack`

Output:
* labeled cells :py:class:`BinaryMaskCollection`
"""

# First we need to filter and register in situ sequencing (ISS) data.

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from showit import image
from starfish.image import ApplyTransform, LearnTransform
from starfish.image import Filter
from starfish.types import Axes

from starfish import data, FieldOfView

matplotlib.rcParams["figure.dpi"] = 150

use_test_data = os.getenv("USE_TEST_DATA") is not None
experiment = data.ISS(use_test_data=use_test_data)
fov = experiment.fov()

imgs = fov.get_image(FieldOfView.PRIMARY_IMAGES) # primary images
dots = fov.get_image("dots") # reference round for image registration
nuclei = fov.get_image("nuclei") # nuclei`

# filter raw data
masking_radius = 15
filt = Filter.WhiteTophat(masking_radius, is_volume=False)
filt.run(imgs, in_place=True)
filt.run(dots, in_place=True)
filt.run(nuclei, in_place=True)

# register primary images to reference round
learn_translation = LearnTransform.Translation(reference_stack=dots, axes=Axes.ROUND, upsampling=1000)
transforms_list = learn_translation.run(imgs.reduce({Axes.CH, Axes.ZPLANE}, func="max"))
warp = ApplyTransform.Warp()
warp.run(imgs, transforms_list=transforms_list, in_place=True)

###################################################################################################
# BespokeWatershed
# ================
#
# BespokeWatershed allows the user to customize the watershed segmentation workflow. This tutorial
# will replicate the EZWatershed algorithm. There are two additional parameters to set, which were
# unchangeable in EZWatershed:
#
# Min_allowed_size: minimum size (pixels) of nuclei to be used as markers for watershed
#
# Max_allowed_size: maximum size (pixels) of nuclei to be used as markers for watershed

from starfish.morphology import Binarize, Filter, Merge, Segment
from starfish.types import Levels

dapi_thresh = .18  # binary mask for cell (nuclear) locations
stain_thresh = .22  # binary mask for overall cells // binarization of stain
min_dist = 57
min_allowed_size = 10
max_allowed_size = 10000

###################################################################################################
# The first step is to maximum project and scale the primary and nuclei images. The primary images
# are treated as a stain for the whole cell, which can be segmented. The nuclei image is used to
# find markers that seed the watershed segmentation of the stain.
#
# The projected and scaled images shown below can be useful for choosing thresholds.

mp = imgs.reduce({Axes.CH, Axes.ZPLANE}, func="max")
stain = mp.reduce(
    {Axes.ROUND},
    func="mean",
    level_method=Levels.SCALE_BY_IMAGE)

nuclei_mp_scaled = nuclei.reduce(
    {Axes.ROUND, Axes.CH, Axes.ZPLANE},
    func="max",
    level_method=Levels.SCALE_BY_IMAGE)

f = plt.figure(figsize=(12,5))
ax1 = f.add_subplot(121)
nuclei_numpy = nuclei_mp_scaled._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)
image(nuclei_numpy, ax=ax1, size=20, bar=True)
plt.title('Nuclei')

ax2 = f.add_subplot(122)
image(
    stain._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE),
    ax=ax2, size=20, bar=True)
plt.title('Stain')

####################################################################################################
# Next we binarize and segment the nuclei. MinDistanceLabel segments the nuclei images, using
# watershed to separate nuclei. Segmented nuclei are then filtered by area. The resulting nuclei
# will be markers used to seed the watershed of primary images in the next step.
#
# Check how the dapi threshold binarized the nuclei image on the left and the markers that will
# seed the watershed.

binarized_nuclei = Binarize.ThresholdBinarize(dapi_thresh).run(nuclei_mp_scaled)
labeled_masks = Filter.MinDistanceLabel(min_dist, 1).run(binarized_nuclei)
watershed_markers = Filter.AreaFilter(min_area=min_allowed_size, max_area=max_allowed_size).run(labeled_masks)

plt.subplot(121)
image(
    binarized_nuclei.uncropped_mask(0).squeeze(Axes.ZPLANE.value).values,
    bar=False,
    ax=plt.gca(),
)
plt.title('Nuclei Thresholded')

plt.subplot(122)
image(
    watershed_markers.to_label_image().xarray.squeeze(Axes.ZPLANE.value).values,
    size=20,
    cmap=plt.cm.nipy_spectral,
    ax=plt.gca(),
)
plt.title('Found: {} cells'.format(len(watershed_markers)))

####################################################################################################
# The stain image is binarized and then the union of this thresholded stain and markers from the
# previous step is used to create a mask for watershed.

thresholded_stain = Binarize.ThresholdBinarize(stain_thresh).run(stain)
markers_and_stain = Merge.SimpleMerge().run([thresholded_stain, watershed_markers])
watershed_mask = Filter.Reduce(
    "logical_or",
    lambda shape: np.zeros(shape=shape, dtype=np.bool)
).run(markers_and_stain)

image(
    watershed_mask.to_label_image().xarray.squeeze(Axes.ZPLANE.value).values,
    bar=False,
    ax=plt.gca(),
)
plt.title('Watershed Mask')

#####################################################################################################
# The final step is to run seeded watershed segmentation on the stain image using markers from
# segmenting nuclei and the mask from merging nuclei and stain together.

segmenter = Segment.WatershedSegment(connectivity=np.ones((1, 3, 3), dtype=np.bool))
masks = segmenter.run(
    stain,
    watershed_markers,
    watershed_mask,
)

#####################################################################################################
# The result is displayed below.

image(
    masks.to_label_image().xarray.squeeze(Axes.ZPLANE.value).values,
    size=20,
    cmap=plt.cm.nipy_spectral,
    ax=plt.gca(),
)
plt.title('Segmented Cells')

###################################################################################################
# EZWatershed
# ===========
#
# EZWatershed segmentation only requires the nuclei imagestack, the primary imagestack, and three
# parameters to be set:
#
# Nuclei_threshold: float between 0 and 1 to binarize scaled max projected nuclei images
#
# Input_threshold: float between 0 and 1 to binarize scaled max projected primary images
#
# Min_distance: positive int that determines the nearest (in pixels) two nuclei centroids can be
# to be considered separate cells



#####################################################################################################

# TODO mattcai clarify and add description of procedure
# TODO mattcai add links
# TODO mattcai BONUS: improve EZWatershed by binary open operation of watershed mask?
