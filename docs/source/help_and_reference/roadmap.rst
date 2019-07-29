Roadmap
==================

starfish is an open-source Python package that enables researchers using in situ
transcriptomics techniques to process their multi-terabyte imaging datasets,
generating single-cell gene expression matrices with spatial information.

- Support for most in situ hybridization (smFISH, MERFISH, seqFISH, etc.)
  and in situ sequencing (STARMAP, DARTFISH, BARRISTASEQ, etc) assays
- Implements a next-generation file format amenable to distributed cloud-based
  processing of images
- Defines common data processing components for extracting the location and
  identity of individual mRNA molecules from raw images and building cell x gene
  expression matrices annotated with spatial coordinates of each cell

This document describes the work we have prioritized for the coming months. If
you have questions or feedback about this roadmap, please submit an issue on
GitHub.

*Please note: this roadmap is subject to change.*

Last updated: July 17, 2019

What we are building now
------------------------

Image segmentation improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, starfish implements a watershed algorithm to enable segmenting cells,
however, the current implementation is somewhat limited. We plan to improve the
watershed pipeline component by enabling users to select additional parameters
and implement support for 3D segmentation, accept hand-drawn (manual) polygons
from software such as FIJI, and apply pre-trained Ilastik models. See issue
#1321 for more information.

Affine image registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starfish currently supports image registration through linear transformations.
Once scikit-image implements affine image registration (see
`scikit-image/scikit-image PR #4023 <https://github.com/scikit-image/scikit-image/pull/4023>`_), we plan to implement this in starfish. See
issue `#1320 <https://github.com/spacetx/starfish/issues/1320>`_ for more information.

Unification of spot finding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We plan to standardize the common steps in various spot finding approaches, so
that they can be used in coordination to achieve many different ways of
detecting, locating, grouping, decoding, and filtering spot data for both
multiplexed and non-multiplexed assays. See Epic `#1450 <https://github.com/spacetx/starfish/issues/1450>`_ for more information.

Improve usability & robustness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a number of areas where we seek to improve the durability of the
package for users. These include improving logging and versioning file formats.
See issue `#1309 <https://github.com/spacetx/starfish/issues/1309>`_ for more information.

What we are building later
---------------------------

Improvements to combining FOVs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Work to date has focused on processing individual fields of view, with limited
support for combining results across fields of view. Depending on the specific
needs of our users, we plan to improve the ability to merge data from multiple
fields of view, including spots, cells, gene expression matrixes, and images.

Quality Control Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We’ve begun to explore assay-agnostic quality control metrics that can be used
to assess the quality of an experiment. We plan to add support for these
metrics. See issue `#61 <https://github.com/spacetx/starfish/issues/61>`_ for details.

Improved support for Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, starfish works only on Windows 10 in the WSL. In the future, we plan
to implement Windows-native support. See issue `#1296 <https://github.com/spacetx/starfish/issues/1296>`_ for details.
