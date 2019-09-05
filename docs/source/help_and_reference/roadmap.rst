.. _roadmap:

Roadmap
=======

starfish is an open-source Python package that enables researchers using in situ transcriptomics techniques to
process multi-terabyte image-based transcriptomics datasets, generating single-cell gene expression matrices with
spatial information.  These are the features we believe will deliver value to scientists piloting image-based
transcriptomics assays or using them at scale:

* Support for most in-situ hybridization (smFISH, MERFISH, seqFISH) and in-situ sequencing (STARMAP, DARTFISH,
  BARRISTASEQ, etc) assays, to enable easy switching between different chemistries.

* Support of a next-generation chunked file format amenable to distributed processing of images from multiple
  fields of view, to enable fast, parallel processing of large volumes of image data.

* Common data processing components for extracting the location and identity of individual mRNA molecules
  from raw images and building cell x gene expression matrices annotated with spatial coordinates of each cell, to
  enable comparative analysis between different chemistries or imaging approaches.

This document describes the two major foci of starfish for the coming months that we believe will deliver the most
value to our users, followed by work we believe is important but that is not yet prioritized. If you have questions or
feedback about this roadmap, please submit an issue on GitHub. If there is work not on our roadmap that is important
to your project, please consider :ref:`contributing`.

*Please note: this roadmap is subject to change.*

Last updated: August 29, 2019

Core Algorithm Functionality
----------------------------
The first stage of starfish development implemented the minimum set of methods needed to ensure that starfish matches
the spot calling results from published analyses of the image-based transcriptomics assays developed by SpaceTx labs.
However, this came at the expense of upstream features such as stitching and registration, downstream features like
segmentation, general ease of use, and documentation. Improvements to each of these are necessary to enable users
who are not themselves image analysis experts to process image-based transcriptomics data.

Image segmentation improvements `#1500 <https://github.com/spacetx/starfish/issues/1500>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Currently, starfish implements a watershed algorithm to enable segmenting cells, however, some of our users find our
implementation difficult to use. We've noticed that as a result, our users call spots and then use
external tools to carry out segmentation. We plan to simplify the watershed method and enable support for 3D
segmentation, accept hand-drawn polygons from software such as FIJI, and apply pre-trained Ilastik models.
We will add vignettes that demonstrate how these segmentation approaches can be used, and where applicable, how to
tune parameters to effectively segment cells in different assays and tissue types.

Affine image registration `#1320 <https://github.com/spacetx/starfish/issues/1320>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starfish currently supports image registration through linear translations. However, many image acquisition
strategies require more sophisticated transformations to align spots prior to decoding. Once scikit-image implements
affine image registration
(see `scikit-image/scikit-image PR #4023 <https://github.com/scikit-image/scikit-image/pull/4023>`_),
we plan to support this in starfish.

Unification of spot finding `#1450 <https://github.com/spacetx/starfish/issues/1450>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To improve our user's ability to compare different spot finding appraoches, we will standardize them so each can be
applied to both multiplexed and non-multiplexed assays, and for both per-round and aggregated data flows. We will add
vignettes that demonstrate how these spot finders can be used, and how to tune parameters to effectively detect spots
in a variety of image signal-to-noise regimes.

Data flow & Usability
---------------------
In addition to ensuring that starfish has the necessary features to support our users, we believe the following work
will make starfish easier and more enjoyable to use.

Windows support `#1296 <https://github.com/spacetx/starfish/issues/1296>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Over half of the scientists we've spoken to would prefer to use Windows for their image processing.
While starfish can be run on Windows machines through the Windows Subsystem for Linux, this feature is not available on
older versions of Windows or Windows Servers, and requires knowledge of Linux, which not all image-based
transcriptomics data analysts may have. We will enable native Windows support so that users can run starfish
on their machine of choice directly.

Multi Field of View Workflows `#1338 <https://github.com/spacetx/starfish/issues/1338>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starfish was originally built with the expectation that users would pre-register their data such that the positions of
images in global physical space have been pre-corrected. We recognize that this results in duplicative data processing
for some data generation patterns, which adds a significant speed penalty to starfish. To address this, we will
implement a registration workflow that identifies and stores optimal transforms for each image, such that spots
and cells extracted from the images have correct global positions. Starfish will not (yet) support applying the
transforms and resolving overlapping regions to produce fused image collages.

Simplified Documentation `#1522 <https://github.com/spacetx/starfish/issues/1522>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Many users question how to format data for starfish, and how to tune parameters for image correction, spot finding, and
segmentation. To make starfish faster to set up, and to clarify how to create an image processing workflow for
non-experts, Starfish will document how to set up a workflow for formatting data into SpaceTx format from common data
formats, processing sequential, coded, or mixed experiments, tune parameters for each of its algorithms, and carry out
quality control checks to verify that starfish is producing accurate results.

What's next for starfish?
-------------------------
The following work is not currently prioritized, but we understand its importance and intend to
address these areas after the above work is completed.

Speed
~~~~~
Image processing, particularly of volumetric data, is a time intensive process. Starfish will explore support for
parallelism on HPC clusters and support for GPU computation in starfish, such that future work or enterprising users
can add GPU-enabled methods to starfish. Finally, we will create benchmarks to track the speed of starfish operations
to catch performance regressions.

Resolving overlaps between fields of view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Work to date has focused on processing individual fields of view, with functional but limited
support for combining results across fields of view. We plan to improve the ability to merge
features, such as spots or cells, that overlap multiple fields of view.

Durability & Versioning `#1309 <https://github.com/spacetx/starfish/issues/1309>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are a number of areas where we seek to improve the durability of the package for users. These include improving
logging and versioning file formats, so that it is always clear how a given output was produced and how to reproduce it.

Quality Control `#61 <https://github.com/spacetx/starfish/issues/61>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We’ve begun to explore assay-agnostic quality control metrics that can be used to assess the quality of an experiment.
We plan to add support for these metrics so our users are better equippied to evaluate the quality of their data and
data processing routines.

Simplifying Contribution Patterns `#1521 <https://github.com/spacetx/starfish/issues/1521>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We are very interested in contribution of algorithms in active research areas, such as segmentation, or spot decoding.
Therefore, we want to make starfish easy and fun to contribute to. We recognize that parts of starfish, by virtue of the
multi-field of view data flow, are quite complex. We will endeavor to simplify the package and clarify API documentation
to streamline future contribution. We've opened an issue to source points of confusion. Comments on this issue will help
us identify places where our development team can focus our efforts.

Proteomics Support
~~~~~~~~~~~~~~~~~~
Proteomics assays follow similar data analysis patterns to RNA-based assays, but have some unique requirements that we
expect to require additional algorithms. Starfish would like to add support for these assays.
