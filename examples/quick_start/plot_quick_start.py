"""
.. _quick start:

Quick Start
===========

The purpose of this hands-on tutorial is to acquaint first-time users to all the basic components
of a starfish pipeline. Anyone with a computer and internet connection should be able to follow
the guide to create and run the example starfish pipeline on the provided experiment data.

Approximate time to complete tutorial: 10 minutes

Prerequisites
-------------

* Python 3.6+ installed (Python 3.7 recommended)
* Some basic knowledge of scientific computing in Python_
* :ref:`Starfish library installed <installation>`
* seaborn_ is installed: :code:`pip install seaborn`

.. _Python: https://docs.scipy.org/doc/numpy/user/quickstart.html
.. _seaborn: https://seaborn.pydata.org/

Starfish Pipeline
-----------------

Open a Jupyter notebook or console with IPython. Make sure you're using the kernel or virtual
environment where you installed starfish, and then import the following modules:
"""

####################################################################################################
# **Load example dataset**
#
# For this tutorial, we use one of the datasets included in the starfish library. It is one
# :term:`FOV<Field of View (FOV)>` from a co-cultured mouse and human fibroblast *in situ*
# sequencing (ISS) experiment. The dataset contains primary images and auxiliary images. The
# primary images consist of 4 rounds that sequence 4 bases of the beta-actin (ACTB) gene. The
# mouse and human ACTB genes differ by one base, allowing mouse and human cells to be labeled.
# The auxiliary images consist of a "dots" image, which is an image containing all
# :term:`rolonies<Rolony>` stained in one channel, and a dapi image. This data was published in
# `Ke, et al. <https://www.ncbi.nlm.nih.gov/pubmed/23852452>`_

from starfish import data

# Download experiment
e = data.MOUSE_V_HUMAN()
# Load ImageStacks from experiment
imgs = e.fov().get_image('primary')  # primary images consisting of 4 rounds and 4 channels
dots = e.fov().get_image('dots')  # auxiliary image with all rolonies stained
nuclei = e.fov().get_image('nuclei')  # auxiliary image with dapi stain

####################################################################################################
# **Confirm data is loaded**
#
# Evaluating an :ref:`ImageStack` expression will return the shape of the `ImageStack` (a
# 5-Dimensional tensor). In this example, the primary images consist of 4 rounds, 4 channels, 1
# z-plane, 980 y-pixels, and 1330 x-pixels.

imgs

####################################################################################################
# **Interactively visualize primary images**
#
# Using :py:func:`.display` you can visualize `ImageStacks` in *napari*. This is a very
# handy tool to examine your data and can help you  make pipeline building decisions. If you use
# the dimension slider to compare channels 1 and 3 of round 1 (zero-based indexing),
# you can already see which cells express mouse ACTB and which cells express human ACTB.
#
# .. code-block:: python
#
#   from starfish import display
#
#   %gui qt
#   viewer = display(imgs)
#   viewer.layers[0].name = "raw stack" # rename the layer
#
# .. image:: /_static/images/quickstart-napari-screenshot.png

####################################################################################################
# **View codebook**
#
# This codebook is an example of a *one hot exponentially multiplexed* codebook where every round
# is "one hot", meaning every round has exactly one channel with signal. This codebook has two
# codewords, which are represented by the 2-Dimensional arrays and are labeled by the gene in the
# "target" coordinate. The "r" (rounds) and "c" (channels) correspond to the rows and columns,
# respectively. The size of the codeword arrays should equal the "r" and "c" dimensions of the
# primary imagestack.

e.codebook

####################################################################################################
# **Define functions that make up the image processing pipeline**
#
# To keep the quick start tutorial quick, only brief descriptions of each function are included
# here. In-depth discussion of each component can be found in the :ref:`User Guide
# <user_guide>` and :ref:`API` documentation.
#
# *Image Registration:*
#
# Translational shifting between rounds of imaging can be corrected for by image registration.
# For this data, the transforms are learned from the "dots" auxiliary image and then applied to
# the primary images.


from starfish.image import ApplyTransform, LearnTransform
from starfish.types import Axes


def register(imgs, dots, method = 'translation'):
    mip_imgs = imgs.reduce(dims = [Axes.CH, Axes.ZPLANE], func="max")
    mip_dots = dots.reduce(dims = [Axes.CH, Axes.ZPLANE], func="max")
    learn_translation = LearnTransform.Translation(reference_stack=mip_dots, axes=Axes.ROUND, upsampling=1000)
    transforms_list = learn_translation.run(mip_imgs)
    warp = ApplyTransform.Warp()
    registered_imgs = warp.run(imgs, transforms_list=transforms_list, in_place=False, verbose=True)
    return registered_imgs

####################################################################################################
# *Image Filtering:*
#
# Depending on the sample, there can be significant background autofluorescence in the image.
# Clusters of background, with radius greater than expected rolony radius can be removed with a
# white tophat filter.


from starfish.image import Filter


def filter_white_tophat(imgs, dots, masking_radius):
    wth = Filter.WhiteTophat(masking_radius=masking_radius)
    return wth.run(imgs), wth.run(dots)

####################################################################################################
# *Finding Spots:*
#
# There are many approaches and algorithms for finding spots in an image. For this pipeline,
# spots are found in the "dots" image using a laplacian-of-gaussian blob detection algorithm.
# Then an :ref:`IntensityTable` is constructed by measuring the pixel intensities at the spot
# locations in each primary image.


from starfish.spots import FindSpots


def find_spots(imgs, dots):

    p = FindSpots.BlobDetector(
        min_sigma=1,
        max_sigma=10,
        num_sigma=30,
        threshold=0.01,
        measurement_type='mean',
    )

    intensities = p.run(image_stack=imgs, reference_image=dots)
    return intensities

####################################################################################################
# *Decoding Spots:*
#
# The method of decoding spots in an :ref:`IntensityTable` depends on the experimental design
# that is reflected in the codebook. Since this example uses a *one hot exponentially
# multiplexed* codebook, :py:class:`.PerRoundMaxChannel` is a logical choice for decoding. It
# decodes the spot by taking the channel with max intensity in each round to construct a barcode,
# which is then matched to a codeword in the codebook. In the case of *in situ sequencing*,
# the barcode is actually the sequence of the gene.


from starfish.spots import DecodeSpots


def decode_spots(codebook, spots):
    decoder = DecodeSpots.PerRoundMaxChannel(codebook=codebook)
    return decoder.run(spots=spots)

####################################################################################################
# *Segmenting Cells:*
#
# Segmenting the cells in this FOV is done by thresholding and watershed segmentation. The
# dapi-stained nuclei are first thresholded and watershed segmented to act as seeds for watershed
# segmentation of the cell stain image. For this dataset, the cell stain image is the
# maximum-intensity projection of primary images, which has autofluorescence inside the cell
# and very little background outside the cell. There are also :ref:`other cell segmentation
# methods <section_segmenting_cells>` available in starfish.


import numpy as np
from starfish.image import Segment
from starfish.types import Axes


def segment(registered_imgs, nuclei):
    dapi_thresh = .22  # binary mask for cell (nuclear) locations
    stain_thresh = 0.011  # binary mask for overall cells // binarization of stain
    min_dist = 56

    registered_mp = registered_imgs.reduce(dims=[Axes.CH, Axes.ZPLANE], func="max").xarray.squeeze()
    stain = np.mean(registered_mp, axis=0)
    stain = stain / stain.max()
    nuclei = nuclei.reduce(dims=[Axes.ROUND, Axes.CH, Axes.ZPLANE], func="max")

    seg = Segment.Watershed(
        nuclei_threshold=dapi_thresh,
        input_threshold=stain_thresh,
        min_distance=min_dist
    )
    masks = seg.run(registered_imgs, nuclei)

    return seg, masks

####################################################################################################
# *Single Cell Gene Expression Matrix:*
#
# A single cell gene expression matrix is made by first labeling spots with the cell ID of the cell
# that the spot is located in. Then the :py:class:`DecodedIntensityTable` of spots is transformed
# into an expression matrix by counting the number of spots for each gene in each cell.

from starfish.spots import AssignTargets


def make_expression_matrix(masks, decoded):
    al = AssignTargets.Label()
    labeled = al.run(masks, decoded[decoded.target != 'nan'])
    cg = labeled[labeled.cell_id != 'nan'].to_expression_matrix()
    return cg

####################################################################################################
# **Run pipeline**
#
# Now that every function of the pipeline has been defined, we can run the pipeline on *this*
# FOV with just a few lines of code. For production scale image processing, this pipeline can be
# run on multiple FOVs in parallel.

# filter
imgs_wth, dots_wth = filter_white_tophat(imgs, dots, 15)

# register
registered_imgs = register(imgs_wth, dots_wth)

# find spots
spots = find_spots(registered_imgs, dots_wth)

# decode
decoded = decode_spots(e.codebook, spots)

# segment
seg, masks = segment(registered_imgs, nuclei)

# make expression matrix
mat = make_expression_matrix(masks, decoded)

####################################################################################################
# **Number of spots found**

spots.count_total_spots()

####################################################################################################
# **Visualize spots and masks on primary images**
#
# You can now visualize the primary images (:py:class:`.ImageStack`),
# decoded spots (:py:class:`.DecodedIntensityTable`),
# and segmented cells (:py:class:`.BinaryMaskCollection`) as layers in napari to verify the
# results.
#
# While the accuracy of the segmentation on the left of the FOV may seem suspicious (is that one cell
# or two?), exploring the primary images and decoded spots demonstrates that there are two cells: one
# expressing the human ACTB gene and one expressing the mouse.
#
# .. code-block:: python
#
#   display(stack=registered_imgs, spots=decoded, masks=masks, viewer=viewer)
#
# .. image:: /_static/images/quickstart-napari-screenshot-2.png

####################################################################################################
# **View decoded spots as a table**

print(decoded.to_features_dataframe().head(10))

####################################################################################################
# **Number of mouse and human ACTB spots that pass threshold**
#
# An advantage of sequencing or barcoding transcripts in situ is the ability to use the multiple
# rounds of image acquisition as a way to error check. For example, false positives from the
# spot finding step are removed if the pixel intensities of the spot across all rounds of images
# don't match a codeword in the codebook.

import numpy as np
import pandas as pd
from starfish.types import Features

genes, counts = np.unique(decoded.loc[decoded[Features.PASSES_THRESHOLDS]][Features.TARGET], return_counts=True)
table = pd.Series(counts, index=genes).sort_values(ascending=False)
print(table)

####################################################################################################
# **View single cell gene expression matrix as a table**

print(mat.to_pandas())

####################################################################################################
# **View single cell gene expression matrix as a heatmap**
#
# The heatmap makes it very clear that based on gene expression, three cells (cells 1, 3,
# and 4) are mouse cells and two (cells 2 and 5) are human cells.

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30, 10))
sns.set(font_scale=4)
sns.heatmap(mat.data.T,
            yticklabels=['mouse', 'human'],
            xticklabels = ['cell {}'.format(n+1) for n in range(5)],
            cmap='magma')

####################################################################################################
# This is the end of the quick start tutorial! To use starfish on your own data, start with
# :ref:`Data Formatting<section_formatting_data>` and then follow the :ref:`User
# Guide<user_guide>` to create a pipeline tailored to your data. If
# you want to try creating pipelines but don't have data yet, :py:mod:`starfish.data` has a
# number of example datasets you can experiment with.
