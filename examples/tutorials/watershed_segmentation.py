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
first is to use :py:class:`.WatershedSegment` to explicitly define a
segmentation pipeline. The second is to use :py:class:`.Watershed`, a predefined watershed
segmentation pipeline that uses watershed segmented nuclei as seeds to run
watershed segmentation on cell stain images.

The predefined watershed segmentation pipeline will not work for all data, so this tutorial
will first show you how you can replicate the predefined watershed segmentation pipeline using the
classes and methods provided in :py:mod:`.morphology`. Then this tutorial will cover how to run
the predefined segmentation pipeline.

Inputs for this tutorial are :py:class:`.ImageStack`\s:

* registered primary images to mimic a cell stain
* registered nuclei images to seed the water segmentation of cells

Output is a :py:class:`.BinaryMaskCollection`:

* each binary mask in the collection is a labeled cell
"""

# First we need to create our inputs by filtering and registering in situ sequencing (ISS) data.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from showit import image
from starfish.image import ApplyTransform, LearnTransform, Filter
from starfish.types import Axes, Levels
from starfish import data, FieldOfView

matplotlib.rcParams["figure.dpi"] = 150

experiment = data.ISS()
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
# Example of Custom Watershed Pipeline
# ====================================
#
# Starfish allows the user to build a custom watershed segmentation pipeline. This tutorial
# will demonstrate how construct a pipeline by replicating the algorithm behind starfish's
# :py:class:`.Watershed` segmentation.
#
# In order to use this algorithm, the :term:`primary images<Primary Images>` must have higher
# background intensity in intracellular regions than extracellular regions such that the
# intracellular regions can be labeled foreground by thresholding. This is not always the case
# since it is usually beneficial to tune the microscope parameters to minimize background and
# increase the SNR of spots in the acquired primary images. :term:`Auxiliary images<Auxiliary
# Images>` with cell stains would be ideal for this purpose and should be used instead if
# experimentally feasible.
#
# :term:`Auxiliary images <Auxiliary Images>` with nuclear stains (e.g. DAPI) are also required in
# this algorithm to seed watershed segmentation of the cells. While a distance transformation of
# the binarized cell stain images could also be used to seed watershed, nuclear stained images
# are almost always included in an experiment. The advantage of using nuclei is that it will
# result in a more accurate number of cells since it is not as prone to over-segmentation
# artifacts as using the distance transform and nuclear stains are usually more robust than
# cellular stains.
#
# There are a number of parameters that need to be tuned for optimal segmentation. Generally,
# the same parameters can be used across an experiment unless there is variation in microscope
# settings or tissue characteristics (e.g. autofluorescence).

# import methods for transformations on or to morphological data
from starfish.morphology import Binarize, Filter, Merge, Segment

# set parameters
dapi_thresh = .18  # global threshold value for nuclei images
stain_thresh = .22  # global threshold value for primary images
min_dist = 57  # minimum distance (pixels) between nuclei distance transformed peaks
min_allowed_size = 10  # minimum size (in pixels) of nuclei
max_allowed_size = 10000  # maximum size (in pixels) of nuclei

###################################################################################################
# This segmentation is done on 2D images so :py:class:`.reduce` is used to maximum intensity project
# and scale the primary and nuclei image volumes. Displaying the projected and scaled images can
# be useful for choosing thresholds.

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
# In order to obtain the seeds to watershed segment cells, the nuclei are labeled first.
# :py:class:`.ThresholdBinarize` thresholds and :py:class:`.MinDistanceLabel` labels the
# binarized nuclei by using a distance transform followed by watershed. :py:class:`.AreaFilter`
# then filters nuclei by area. The ``min_dist`` parameter that limits how close two
# peaks in the distance transformed image can be is key to preventing over-segmentation of nuclei
# with non-circular morphology but may also incorrectly merge two nuclei in close proximity.
#
# The binarized and segmented nuclei can be viewed to determine whether parameters need to be
# adjusted.

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
# The cell stain image (i.e. primary image) is binarized and then the union of the binary cell
# image and binary nuclei image is used to create a mask for watershed. This ensures that the
# nuclei markers that seed watershed segmentation of cells are all within cells and that no area
# outside of cells is labeled.

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

####################################################################################################
# The final step is to run seeded watershed segmentation. The pixel values of the cell stain
# define the topology of the watershed basins. The basins are filled starting from seeds,
# which are the nuclei markers. And the boundaries of the basins are the watershed mask.

segmenter = Segment.WatershedSegment(connectivity=np.ones((1, 3, 3), dtype=np.bool))

# masks is BinaryMaskCollection for downstream steps
masks = segmenter.run(
    stain,
    watershed_markers,
    watershed_mask,
)

# display result
image(
    masks.to_label_image().xarray.squeeze(Axes.ZPLANE.value).values,
    size=20,
    cmap=plt.cm.nipy_spectral,
    ax=plt.gca(),
)
plt.title('Segmented Cells')

###################################################################################################
# Pre-defined Watershed Segmentation Pipeline
# ===========================================
#
# Running :py:class:`.Watershed` from the :py:mod:`.image` module (not to be confused with
# :py:class:`.WatershedSegment` from the :py:mod:`.morphology` module) is a convenient method to
# apply the same segmentation algorithm that was built in the previous section of this tutorial.
# It hardcodes the ``min_allowed_size`` and ``max_allowed_size`` of the nuclei to 10
# pixels and 1,000 pixels, respectively, but accepts the other user-defined parameters as arguments.
#
# Here is an example of how to run :py:class:`.Watershed` on the same set of images as the
# previous section. The intermediate results are saved as attributes of the
# :py:class:`.Watershed` instance and can be displayed to assess performance.

from starfish.image import Segment

# set parameters
dapi_thresh = .18  # global threshold value for nuclei images
stain_thresh = .22  # global threshold value for primary images
min_dist = 57  # minimum distance (pixels) between nuclei distance transformed peaks

seg = Segment.Watershed(
    nuclei_threshold=dapi_thresh,
    input_threshold=stain_thresh,
    min_distance=min_dist
)

# masks is BinaryMaskCollection for downstream steps
masks = seg.run(imgs, nuclei)

# display intermediate images and result
seg.show()
