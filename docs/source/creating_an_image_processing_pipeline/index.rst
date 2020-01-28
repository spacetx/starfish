.. _creating_an_image_processing_pipeline:

Creating an Image Processing Pipeline
=====================================

Welcome to the user guide for building an image processing pipeline using starfish! This tutorial
will cover all the steps necessary for going from raw images to a single cell gene expression
matrix. If you are wondering what is starfish, check out :ref:`The Introduction
<introduction>`. If you only have a few minutes to try out starfish, check out a pre-built
pipeline by following the :ref:`Guide to Getting Started<getting_started>`. If you are ready
to learn how to build your own image processing pipeline using starfish then read on!

The :ref:`data model<data_model>`

This part of the tutorial goes into more detail about why each of the stages in the example are
needed, and provides some alternative approaches that can be used to build similar pipelines.

The core functionalities of starfish pipelines are the detection (and decoding) of spots, and the
segmentation of cells. Each of the other approaches are designed to address various characteristics
of the imaging system, or the optical characteristics of the tissue sample being measured, which
might bias the resulting spot calling, decoding, or cell segmentation decisions. Not all parts of
image processing are always needed; some are dependent on the specific characteristics of the
tissues. In addition, not all components are always found in the same order. *Starfish* is flexible
enough to omit some pipeline stages or disorder them, but the typical order might match the
following. The links show how and when to use each component of *starfish*, and the final section
demonstrates putting together a "pipeline recipe" and running it on an experiment.

Loading Data
------------

* :ref:`Formatting your data <data_conversion_examples>`
* :ref:`Using formatted example data <datasets>`

Manipulating Images
-------------------

Sometimes it can be useful subset the images by, for example, excluding out-of-focus images or
cropping out edge effects. For sparse data, it can be useful to project the z-volume into a single
image, as this produces a much faster processing routine.

* :ref:`Cropping <tutorial_cropping>`
* :ref:`Projecting <tutorial_projection>`

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

Enhancing Signal & Removing Background Noise
--------------------------------------------

These stages are usually specific to the sample being analyzed. For example, tissues often have
some level of autofluorescence which causes cellular compartments to have more background noise than
intracellular regions. This can confound spot finders, which look for local intensity differences.
These approaches ameliorate these problems.

* :ref:`Removing Autofluorescence <tutorial_removing_autoflourescence>`

Normalizing Intensities
-----------------------

Most assays are designed such that intensities need to be compared between rounds and/or channels
in order to decode spots. As a basic example, smFISH spots are labeled by the channel with the
highest intensity value. But because different channels use different fluorophores, excitation
sources, etc. the images have different ranges of intensity values. The background
intensity values in one channel might be as high as the signal intensity values of a
different channel. Normalizing the intensities corrects for these differences and allows
comparisons to be made.

Whether to normalize
^^^^^^^^^^^^^^^^^^^^

The decision of whether to normalize depends on your data, codebook schema, and decoding method
used in the next step of the pipeline.
If your images have good SNR with similar range of intensities across channels and you plan to
use :py:class:`PerRoundMaxChannel`, normalizing may not be necessary.
If you plan to decode spots with :py:class:`MetricDistance` or :py:class:`PixelSpotDecoder`, you
*need* to normalize across channels and rounds to get accurate results.
:ref:`Plotting intensity distributions<tutorial_intensity_histogram>` of the
:py:class:`ImageStack` can help you determine whether and how to normalize.

How to normalize
^^^^^^^^^^^^^^^^

How to normalize depends on your data and a key assumption. If you are confident that image
volumes acquired for every channel and/or every round should have the same distribution of
intensities (meaning the number of spots and amount of background autofluorescence in every image
volume is approximately uniform across channels and/or rounds), then their intensity *distributions*
can be normalized with :py:class:`MatchHistograms`. However in most cases this is not a valid
assumption and you can use :py:class:`Clip`, :py:class:`ClipPercentileToZero`, and
:py:class:`ClipValueToZero` to normalize intensity *values*.

Tutorials for normalizing:

* :ref:`Normalizing Intensity Distributions <tutorial_normalizing_intensity_distributions>`
* :ref:`Normalizing Intensity Values <tutorial_normalizing_intensity_values>`

Finding and Decoding Spots
--------------------------

Segmenting Cells
----------------

Assigning Spots to Cells
------------------------

Assessing Performance Metrics
-----------------------------

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

