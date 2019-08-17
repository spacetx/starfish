Roadmap
=======

starfish is an open-source Python package that enables researchers using *in situ*
transcriptomics techniques to process multi-terabyte imaging datasets,
generating single-cell gene expression matrices with spatial information.

- Support for most *in-situ* hybridization (smFISH, MERFISH, seqFISH)
  and *in-situ* sequencing (STARMAP, DARTFISH, BARRISTASEQ, etc) assays
- Implements a next-generation chunked file format amenable to distributed cloud-based
  processing of images
- Defines common data processing components for extracting the location and
  identity of individual mRNA molecules from raw images and building cell x gene
  expression matrices annotated with spatial coordinates of each cell

This document describes the two major foci of starfish for the coming months, followed by work that
is not yet prioritized. If you have questions or feedback about this roadmap, please submit an issue
on GitHub. If there is work not on our roadmap that is important to your project, please consider
:ref:`contributing`.

*Please note: this roadmap is subject to change.*

Last updated: August 17, 2019

Core Algorithm Functionality
----------------------------
The first stage of starfish development implemented the minimum set of methods needed to ensure that
starfish matches the spot calling results from published analysis for the main image-based
transcriptomics assays. However, this came at the expense of downstream methods like segmentation,
general ease of use, and documentation, which we will now improve.

Image segmentation improvements `#1321 <https://github.com/spacetx/starfish/issues/1321>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, starfish implements a watershed algorithm to enable segmenting cells,
however, the current implementation is somewhat limited. We plan to improve the
watershed pipeline component by enabling users to select additional parameters
and implement support for 3D segmentation, accept hand-drawn (manual) polygons
from software such as FIJI, and apply pre-trained Ilastik models.

Affine image registration `#1320 <https://github.com/spacetx/starfish/issues/1320>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starfish currently supports image registration through linear transformations. However, we recognize
that many assays require more forms of fine registration to align spots prior to decoding.
Once scikit-image implements affine image registration (see
`scikit-image/scikit-image PR #4023 <https://github.com/scikit-image/scikit-image/pull/4023>`_),
we plan to support this in starfish.

Unification of spot finding `#1450 <https://github.com/spacetx/starfish/issues/1450>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We plan to standardize the common steps in starfish's spot finding approaches, so
that they can be used in coordination to flexibly detect, group, decode, and filter spot data for
both multiplexed and non-multiplexed assays, and for both per-round and aggregated data flows. We
will add vignettes that demonstrate how these spot finders can be used, and how to tune parameters
to effectively detect spots in a variety of image signal-to-noise regimes.

Data flow & Usability
---------------------

Windows support `#1296 <https://github.com/spacetx/starfish/issues/1296>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A large number of starfish users leverage windows for parameter tuning and image processing. While
starfish supports Windows through the Windows Subsystem for Linux, We will
also enable native Windows to enable users to leverage starfish's parallelism and run starfish on
Windows servers.

Multi Field of View Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starfish was originally built with the expectation that users would pre-register their data such
that the positions of images in global physical space have been pre-corrected. We recognize that
this results in duplicative data processing and will implement a registration method that identifies
and stores optimal transforms for each image, such that *spots and cells* extracted from the images
have correct global positions. Starfish will not (yet) support *applying* the transforms and
resolving overlapping regions to produce image collages.

Speed
~~~~~
Starfish should be *fast*. We will ensure that starfish leverages parallelism
on host machines, be they local machines, HPC clusters, or distributed environments. We will add
support for GPU computation in starfish such that future work or enterprising users can add
GPU-enabled methods to starfish. Finally, we will create benchmarks to track the speed of starfish
operations to catch performance regressions.

What's next for starfish?
-------------------------
The following work is not currently prioritized, but we understand its importance and intend to
address these areas after the above work is completed.

Resolving overlaps between fields of view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Work to date has focused on processing individual fields of view, with limited
support for combining results across fields of view. We plan to improve the ability to merge
features, such as spots or cells, that overlap multiple fields of view.

Durability & Versioning `#1309 <https://github.com/spacetx/starfish/issues/1309>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are a number of areas where we seek to improve the durability of the
package for users. These include improving logging and versioning file formats.

Quality Control `#61 <https://github.com/spacetx/starfish/issues/61>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We’ve begun to explore assay-agnostic quality control metrics that can be used
to assess the quality of an experiment. We plan to add support for these
metrics.

Simplifying Starfish Object Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We want to make starfish easy and fun to contribute to. We recognize that parts of starfish, by
virtue of the multi-field of view data flow, are quite complex. We will endeavor to simplify the
package to streamline future contribution. We've opened an issue for discussion of points of
confusion, and comments on this issue will help us identify places where our development team can
focus our efforts.
