.. _creating_an_image_processing_pipeline:

Creating an Image Processing Pipeline
=====================================

Starfish is a package that exposes methods to detect spots and cells, and to aggregate spot counts
for various target molecules, creating cell x gene matrices that retains spatial information.

These tasks have several steps, and earlier the tutorial goes over a canonical example of an image
processing pipeline that measures the expression of some target RNA in a breast cancer tissue slice.
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

#TODO ask kevin to tell me if I've got these in the right order

Basic Image Manipulations
-------------------------

Sometimes it can be useful subset the images by, for example, excluding out-of-focus images or
cropping out edge effects. For sparse data, it can be useful to project the z-volume into a single
image, as this produces a much faster processing routine.

.. toctree::
    :maxdepth: 1

    tutorials/exec_image_manipulations.rst

Imaging Corrections
-------------------

These stages are typically specific to the microscope, camera, filters, chemistry, and any tissue
handling or microfluidices that are involved in capturing the images. These steps are typically
*independent* of the assay. *Starfish* enables the user to design a pipeline that matches their
imaging system

.. toctree::
    :maxdepth: 1

    tutorials/exec_image_corrections.rst

Tissue/Substrate-specific Corrections
-------------------------------------

These stages are usually specific to the sample being analyzed. For example, tissues often have
some level of autofluorescence which causes cellular compartments to have more background noise than
intracellular regions. This can confound spot finders, which look for local intensity differences.
These approaches ameliorate these problems.

.. toctree::
    :maxdepth: 1

    tutorials/exec_tissue_specific_corrections.rst

Feature Identification and Assignment
-------------------------------------

Once images have been corrected for tissue and optical aberrations, spot finding can be run to
turn those spots into features that can be counted up. Separately,
The dots and nuclei images can be segmented to identify the locations where the cells can be found
in the images. Finally, the two sets of features can be combined to assign each spot to its cell of
origin. At this point, it's trivial to create a cell x gene matrix.

.. toctree::
    :maxdepth: 1

    tutorials/exec_feature_identification_and_annotation.rst

Putting Together a Pipeline Recipe and Running it
-------------------------------------------------

# TODO add blurb here.

.. toctree::
    :maxdepth: 1

    tutorials/exec_running_a_pipeline.rst

Old Content not to be deleted yet.
----------------------------------

.. _document: https://docs.google.com/document/d/1IHIngoMKr-Tnft2xOI3Q-5rL3GSX2E3PnJrpsOX5ZWs/edit?usp=sharing

.. image:: /_static/design/pipeline-diagram.png
    :alt: pipeline diagram
