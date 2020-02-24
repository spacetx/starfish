"""
.. _tutorial_ilastik_segmentation:

Using ilastik
======================

Starfish currently has built-in functionality to support `ilastik <https://www.ilastik.org/>`_,
a segmentation toolkit that leverages machine-learning . Ilastik has a Pixel Classification
workflow that performs semantic segmentation of the image, returning probability maps for each
label (e.g. cells).

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

###################################################################################################
# Importing ilastik Probability Map
# ====================================
#
# Starfish allows the user to build a custom watershed segmentation pipeline. This tutorial
# will demonstrate how construct a pipeline by replicating the algorithm behind starfish's
# :py:class:`.Watershed` segmentation.

###################################################################################################
# Calling out to ilastik Pre-trained Classifier
# =============================================
#
# Starfish allows the user to build a custom watershed segmentation pipeline. This tutorial
# will demonstrate how construct a pipeline by replicating the algorithm behind starfish's
# :py:class:`.Watershed` segmentation.