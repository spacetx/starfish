.. _creating_an_image_processing_pipeline:

Creating an Image Processing Pipeline
=====================================

Welcome to the user guide for building an image processing pipeline using starfish! This tutorial
will cover all the steps necessary for going from raw images to a single cell gene expression
matrix. If you are wondering what is starfish, check out :ref:`The Introduction
<introduction>`. If you only have a few minutes to try out starfish, check out a pre-built
pipeline by following the :ref:`Guide to Getting Started<getting started>`. If you are ready
to learn how to build your own image processing pipeline using starfish then read on!

The :ref:`data model<data_model>`

This part of the tutorial goes into more detail about why each of the stages in the example are
needed, and provides some alternative approaches that can be used to build similar pipelines.

The core functionalities of starfish pipelines are the detection (and :term:`decoding<Decoding>`)
of spots, and the segmentation of cells. Each of the other approaches are designed to address
various characteristics of the imaging system, or the optical characteristics of the tissue
sample being measured, which might bias the resulting spot calling, decoding, or cell
segmentation decisions. Not all parts of image processing are always needed; some are dependent
on the specific characteristics of the tissues. In addition, not all components are always found
in the same order. *Starfish* is flexible enough to omit some pipeline stages or disorder them,
but the typical order might match the following. The links show how and when to use each
component of *starfish*, and the final section demonstrates putting together a "pipeline recipe"
and running it on an experiment.

.. _section_loading_data:

Loading Data
------------

* :ref:`Formatting your data <data_conversion_examples>`
* :ref:`Using formatted example data <datasets>`

.. _section_manipulating_images:

Manipulating Images
-------------------

Sometimes it can be useful subset the images by, for example, excluding out-of-focus images or
cropping out edge effects. For sparse data, it can be useful to project the z-volume into a single
image, as this produces a much faster processing routine.

* :ref:`Cropping <tutorial_cropping>`
* :ref:`Projecting <tutorial_projection>`

.. _section_correcting_images:

Correcting Images
-----------------

These stages are typically specific to the microscope, camera, filters, chemistry, and any tissue
handling or microfluidices that are involved in capturing the images. These steps are typically
*independent* of the assay. *Starfish* enables the user to design a pipeline that matches their
imaging system

* :ref:`Illumination Correction <tutorial_illumination_correction>`
* :ref:`Chromatic Aberration <tutorial_chromatic_aberration>`
* :ref:`Deconvolution <tutorial_deconvolution>`
* :ref:`Image Registration <tutorial_image_registration>`
* :ref:`Image Correction Pipeline <tutorial_image_correction_pipeline>`

.. _section_improving_snr:

Enhancing Signal & Removing Background Noise
--------------------------------------------

These stages are usually specific to the sample being analyzed. For example, tissues often have
some level of autofluorescence which causes cellular compartments to have more background noise than
intracellular regions. This can confound spot finders, which look for local intensity differences.
These approaches ameliorate these problems.

* :ref:`Removing Autofluorescence <tutorial_removing_autoflourescence>`

.. _section_normalizing_intensities:

Normalizing Intensities
-----------------------

Most assays are designed such that intensities need to be compared between :term:`rounds<Imaging
Round>` and/or :term:`channels<Channel>` in order to :term:`decode<Decoding>` spots. As a basic
example, smFISH spots are labeled by the channel with the highest intensity value. But because
different channels use different fluorophores, excitation sources, etc. the images have different
ranges of intensity values. The background intensity values in one channel might be as high as
the signal intensity values of another channel. Normalizing image intensities corrects for these
differences and allows comparisons to be made.

Whether to normalize
^^^^^^^^^^^^^^^^^^^^

The decision of whether to normalize depends on your data and decoding method used in the next
step of the pipeline.
If your :py:class:`.ImageStack` has approximately the same
range of intensities across rounds and
channels then normalizing may have a trivial effect on pixel values. Starfish provides utility
functions :ref:`imshow_plane<tutorial_imshow_plane>` and
:ref:`intensity_histogram<tutorial_intensity_histogram>` to visualize images and their intensity
distributions.

Accurately normalized images is important if you plan to decode features with
:py:class:`.MetricDistance` or :py:class:`.PixelSpotDecoder`. These two algorithms use the
:term:`feature trace<Feature (Spot, Pixel) Trace>` to construct a vector whose distance from
other vectors is used decode the feature. Poorly normalized images with some systematic or random
variation in intensity will bias the results of decoding.

However if you decode with :py:class:`.PerRoundMaxChannel`, which only compares intensities
between channels of the same round, precise normalization is not necessary. As long the intensity
values of signal in all three channels are greater than background in all three channels the
features will be decoded correctly.

How to normalize
^^^^^^^^^^^^^^^^

How to normalize depends on your data and a key assumption. There are two approaches for
normalizing images in starfish:

:ref:`Normalizing Intensity Distributions<tutorial_normalizing_intensity_distributions>`

If you know a priori that image volumes acquired for every channel and/or every round should have
the same distribution of intensities then the intensity *distributions* of image volumes can be
normalized with :py:class:`.MatchHistograms`. Typically this means the number of spots and amount of
background autofluorescence in every image volume is approximately uniform across channels and/or
rounds.

:ref:`Normalizing Intensity Values <tutorial_normalizing_intensity_values>`

In most data sets the differences in gene expression leads to too much variation in number of
spots between channels and rounds. Normalizing intensity distributions would incorrectly skew the
intensities. Instead you can use :py:class:`.Clip`, :py:class:`.ClipPercentileToZero`, and
:py:class:`.ClipValueToZero` to normalize intensity *values* by clipping extreme values and
rescaling.

.. _section_finding_and_decoding:

Finding and Decoding Spots
--------------------------

.. _section_segmenting_cells:

Segmenting Cells
----------------

Unlike single-cell RNA sequencing, image-based transcriptomics methods do not physically separate
cells before acquiring RNA information. Therefore in order to characterize cells, the RNA must be
assigned into single cells by partitioning the image volume. Accurate unsupervised cell-segmentation
is an open problem for all biomedical imaging disciplines ranging from digital pathology to
neuroscience.

The challenge of segmenting cells depends on the structural complexity of the sample and quality
of images available. For example a sparse cell mono-layer with a strong cytosol stain would be
trivial to segment but a dense heterogeneous population of cells in 3D tissue with only a DAPI stain
can be impossible to segment perfectly. On the experimental side, selecting good cell stains and
acquiring images with low background will make segmenting a more tractable task.

There are many approaches for segmenting cells from image-based transcriptomics assays. If you do
not know which method to use, a safe bet is to start with classic thresholding and watershed. On
the other hand, if you can afford to manually segment...

Thresholding and Watershed
^^^^^^^^^^^^^^^^^^^^^^^^^^

The traditional method for segmenting cells in fluorescence microscopy images is to threshold the
image into foreground pixels and background pixels and then label connected foreground as
individual cells. Common issues that affect thresholding such as background noise can be corrected
by preprocessing images before thresholding and filtering connected components after. There are
`many automated image thresholding algorithms <https://imagej.net/Thresholding>`_ but currently
starfish requires manually selecting a global threshold value in :py:class:`.ThresholdBinarize`.

When overlapping cells are labeled as one connected component, they are typically segmented by
using a `distance transformation followed by the watershed algorithm <https://www.mathworks
.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation
.html>`_. Watershed is a classic image
processing algorithm for separating objects in images and can be applied to all types of images.
Pairing it with a distance transform is particularly useful for segmenting convex shapes like
cells.

A segmentation pipeline that consists of thresholding, connected component analysis, and watershed
is the simplest and fastest to implement but its accuracy is highly dependent on image quality.
The signal-to-noise ratio of the cell stain must be high enough for minimal errors after
thresholding and binary operations. And the nuclei or cell shapes must be convex to meet the
assumptions of the watershed algorithm or else it will over-segment. Starfish includes the basic
functions to build a watershed segmentation pipeline and a predefined :py:class:`.Watershed`
segmentation class that uses the :term:`primary images<Primary Images>` as the cell stain:

:ref:`Ways to segment by thresholding and watershed in starfish<tutorial_watershed_segmentation>`

Machine Learning Methods
^^^^^^^^^^^^^^^^^^^^^^^^

Ilastik

Manually Defining Cells
^^^^^^^^^^^^^^^^^^^^^^^

The most accurate but time-consuming approach is to manually segment using a tool like ROI
manager in FIJI ImageJ.
Available methods for segmenting cells.

Points to make:
accuracy is determined by comparison to ground truth, which is just manual assessment.
the process of image segmentation:
`segmentation in imagej <https://imagej.net/plugins/index.html#segmentation>`_

.. _section_assigning_spots:

Assigning Spots to Cells
------------------------

.. _section_assessing_metrics:

Assessing Performance Metrics
-----------------------------

.. _section_utilities:

Other Utilities
---------------

Feature Identification and Assignment
-------------------------------------

Once images have been corrected for tissue and optical aberrations, spot finding can be run to
turn those spots into features that can be counted up. Separately,
The dots and nuclei images can be segmented to identify the locations where the cells can be found
in the images. Finally, the two sets of features can be combined to assign each spot to its cell of
origin. At this point, it's trivial to create a cell x gene matrix.

* :ref:`Spot Finding <tutorial_spot_finding>`
* :ref:`Spot Decoding <tutorial_spot_decoding>`
* :ref:`Segmenting Cells <tutorial_segmenting_cells>`
* :ref:`Assigning Spots to Cells <tutorial_assigning_spots_to_cells>`

