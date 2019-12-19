"""
Watershed Segmentation
=======================

In order to create a cell by gene expression matrix from image-based transcriptomics data, RNA spots
must be assigned to cells. One approach is to segment images of stained cellular structures to define
cell boundaries. A problem arises when there are contiguous or partially overlapping cells, as is
often the case in tissue. These clumped cells are mislabeled as one cell by connected component
labeling.

Watershed segmentation can be used to divide clumped cells by drawing lines between them based on their
shape. More details on watershed algorithm can be found
`here <https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html>`_.

Starfish allows users to implement watershed segmentation into their pipelines in two ways: an easy
but inflexible way and a custom bespoke way. This tutorial will start with the EZWatershed method and
then replicate the results of EZWatershed using the bespoke method.

Input requirements:

* registered primary images :py:class:`ImageStack`

* registered nuclei images :py:class:`ImageStack`

Output:

* labeled cells :py:class:`BinaryMaskCollection`

"""

###################################################################################################
# This tutorial will use in situ sequencing (ISS) data that has been filtered and registered.

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from showit import image

from starfish import data, FieldOfView
from starfish.types import Axes, Features, FunctionSource
from starfish.image import Filter
from starfish.image import ApplyTransform, LearnTransform

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

from starfish.image import Segment

dapi_thresh = .18  # binary mask for cell (nuclear) locations
stain_thresh = .22  # binary mask for overall cells // binarization of stain
min_dist = 7

seg = Segment.Watershed(
    nuclei_threshold=dapi_thresh,
    input_threshold=stain_thresh,
    min_distance=min_dist
)
masks = seg.run(imgs, nuclei)
image(
    masks.to_label_image().xarray.squeeze(Axes.ZPLANE.value).values,
    size=20,
    cmap=plt.cm.nipy_spectral,
    ax=plt.gca(),
)
plt.title('Segmented Cells')

###################################################################################################
# A simple command can display the important images created during EZWatershed. See BespokeWatershed tutorial
# to get a breakdown of each image.
seg.show()

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

from starfish.morphology import Binarize, Filter, Merge
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

plt.subplot(121)
nuclei_numpy = nuclei_mp_scaled._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)
image(nuclei_numpy, ax=plt.gca(), size=20, bar=True)
plt.title('Nuclei')

plt.subplot(122)
image(
    stain._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE),
    ax=plt.gca(), size=20, bar=True)
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
markers = Filter.AreaFilter(min_allowed_size, max_allowed_size).run(labeled_masks)

plt.subplot(121)
image(
    binarized_nuclei.uncropped_mask(0).squeeze(Axes.ZPLANE.value).values,
    bar=False,
    ax=plt.gca(),
)
plt.title('Nuclei Thresholded')

plt.subplot(122)
image(
    markers.to_label_image().xarray.squeeze(Axes.ZPLANE.value).values,
    size=20,
    cmap=plt.cm.nipy_spectral,
    ax=plt.gca(),
)
plt.title('Found: {} cells'.format(len(markers)))

####################################################################################################
# The stain image is binarized and then the union of this thresholded stain and markers from the
# previous step is used to create a mask for watershed.

thresholded_stain = Binarize.ThresholdBinarize(stain_thresh).run(stain)
markers_and_stain = Merge.SimpleMerge().run([thresholded_stain, markers])
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

binarizer = Binarize.WatershedBinarize(connectivity=np.ones((1, 3, 3), dtype=np.bool))
masks = binarizer.run(
    stain,
    markers,
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


#####################################################################################################

# TODO mattcai clarify and add description of procedure
# TODO mattcai add links
# TODO mattcai BONUS: improve EZWatershed by binary open operation of watershed mask?