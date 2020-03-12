"""
.. _tutorial_ilastik_segmentation:

Using ilastik in starfish
=========================

In order to create a cell by gene expression matrix from image-based transcriptomics data, RNA
spots must be assigned to cells. Segmenting a microscopy image into single cells is either
time-consuming when done manually or inaccurate when done with thresholding and watershed. The
advent of deep learning models for image segmentation promises a solution that is both fast and
accurate.

Starfish currently has built-in functionality to support `ilastik <https://www.ilastik.org/>`_,
a segmentation toolkit that leverages machine-learning. Ilastik has a `Pixel Classification
workflow <https://www.ilastik.org/documentation/pixelclassification/pixelclassification/>`_
that performs semantic segmentation of the image, returning probability maps for each
label (e.g. cells). To use ilastik in starfish first install ilastik locally and follow the
Pixel Classification workflow using the GUI to train a classifier, which is saved in an ilastik
project file.

There are two options for using ilastik in starfish. The first is running the trained pixel
classifier on images outside of starfish to generate a probability map that can then be loaded in
starfish as an :py:class:`.ImageStack`. The second is to run the pixel classifier from within
starfish by calling out to ilastik and passing it the trained classifier. The results will be the
same.

The probability map is an nd-array with the same size as the input :py:class:`.ImageStack`. The
value of each element in the nd-array equals the probability of the corresponding pixel to belong to
a label. In ilastik, you can train the pixel classifier to predict the probability for multiple
labels such as nuclei, cells, and background. However, starfish will only load the probability
map of the first label so you should use the first label for cells or nuclei only.

This tutorial will first describe how you can run a trained ilastik classifier from within a
starfish process. In the second section this tutorial will demonstrate how to load an ilastik
probability map hdf5 file as an :py:class:`.ImageStack`. The last section transforms the
:py:class:`.ImageStack` into a :py:class:`BinaryMaskCollection` that can be used by
:py:class:`.Label` to assign spots to cells.
"""

###################################################################################################
# Calling Out to ilastik Trained Pixel Classifier
# ===============================================
#
# Running a trained ilastik pixel classifier from within starfish allows a pipeline to take
# advantage of ilastik's machine-learning model without having to process images from every field
# of view, switching to ilastik to run classifier, exporting all the probability maps, and then
# loading the files back into a starfish pipeline.
#
# The requirements for calling out to ilastik from a starfish pipeline is having ilastik
# installed locally and having an ilastik project that contains a trained classifier. The
# classifier can only be run on an :py:class:`ImageStack` with a single round and channel.
# Obviously, the :py:class:`ImageStack` should be an image of the same type that the classifier
# was trained on for expected probability results.

# Load MERFISH data to get dapi ImageStack
import os
import matplotlib
import matplotlib.pyplot as plt

import starfish.data
from starfish.image import Filter
from starfish.types import Axes, Coordinates, Levels

matplotlib.rcParams["figure.dpi"] = 150

experiment = starfish.data.MERFISH()
fov = experiment["fov_000"]
dapi = fov.get_image("nuclei")

# Process dapi images by blurring and clipping
def preprocess(dapi):
    blur = Filter.GaussianLowPass(sigma=5)
    blurred = blur.run(dapi)
    clip = Filter.Clip(p_min=1, p_max=95, level_method=Levels.SCALE_BY_CHUNK)
    clipped = clip.run(blurred)
    return clipped

dapi = preprocess(dapi)

# Need ilastik script and ilastik project with trained classifier to instantiate filter
ilastik_exe_path = os.path.join(os.path.dirname("__file__"), 'run_ilastik.sh')
ilastik_proj_path = os.path.join(os.path.dirname("__file__"), 'merfish_dapi.ilp')
ipp = Filter.IlastikPretrainedProbability(ilastik_executable=ilastik_exe_path, ilastik_project=ilastik_proj_path)

# Run IlastikPretrainedProbability filter to get probabilities of 'Label 1' as ImageStack
# probabilities = ipp.run(stack=dapi)

###################################################################################################
# Loading ilastik Probability Map
# ===============================
#
# If you already have probability maps of your images from ilastik or prefer to use the ilastik
# GUI it is possible to load the exported hdf5 files into starfish. When using the 'Prediction
# Export' panel in ilastik be sure to use ``Source: Probabilities`` and ``Format: hdf5``. If you
# edit the Dataset name then it must also be passed as ``dataset_name`` to
# :py:meth:`.import_ilastik_probabilities`. The example below loads the same probability map that
# would have been generated in the first section of this tutorial.

# Need to instantiate Filter even though it is not run
ilastik_exe_path = os.path.join(os.path.dirname("__file__"), 'run_ilastik.sh')
ilastik_proj_path = os.path.join(os.path.dirname("__file__"), 'merfish_dapi.ilp')
ipp = Filter.IlastikPretrainedProbability(ilastik_executable=ilastik_exe_path, ilastik_project=ilastik_proj_path)

# Load hdf5 file as ImageStack
h5_file_path = os.path.join(os.path.dirname("__file__"), 'dapi_Probabilities.h5')
imported_probabilities = ipp.import_ilastik_probabilities(path_to_h5_file=h5_file_path)

# View probability map next to dapi image
f, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(dapi.xarray.values.squeeze())
ax1.set_title("Dapi")
ax2.imshow(imported_probabilities.xarray.values.squeeze())
ax2.set_title("Probabilities")
f.tight_layout()

###################################################################################################
# Transforming Probability Map to Masks
# =====================================
#
# In order to use ilastik semantic segmentation probabilities to assign spots to cells in
# starfish they must be transformed into binary cell masks. A straightforward approach to
# transforming the :py:class:`.ImageStack` to a :py:class:`.BinaryMaskCollection` is to threshold
# the probabilities, run connected component analysis, and watershed. This process is analogous to
# segmenting a stained cell image.

# Scale probability map
clip = Filter.Clip(p_min=0, p_max=100, level_method=Levels.SCALE_BY_CHUNK)
clipped_probabilities = clip.run(imported_probabilities)

# Threshold, connected components, watershed, and filter by area
from starfish.morphology import Binarize, Filter
prob_thresh = 0.05
min_dist = 100
min_allowed_size = 5000
max_allowed_size = 100000
binarized_probabilities = Binarize.ThresholdBinarize(prob_thresh).run(clipped_probabilities)
labeled_masks = Filter.MinDistanceLabel(min_dist, 1).run(binarized_probabilities)
masks = Filter.AreaFilter(min_area=min_allowed_size, max_area=max_allowed_size).run(labeled_masks)

# Show probability map along with cell masks
f, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(imported_probabilities.xarray.values.squeeze())
ax1.set_title("Probabilities")
ax2.imshow(labeled_masks.to_label_image().xarray.values.squeeze(), cmap=plt.cm.nipy_spectral)
ax2.set_title("Masks")
f.tight_layout()
