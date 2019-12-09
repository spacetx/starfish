.. _roadmap:

Roadmap
=======

starfish is an open-source Python package that enables researchers using in situ transcriptomics techniques to process their multi-terabyte imaging datasets, generating single-cell gene expression matrices with spatial information.

* Support for most in-situ hybridization (smFISH, MERFISH, seqFISH) and in-situ sequencing (STARMAP, DARTFISH, BARRISTASEQ, etc) assays

* Defines common data processing components for extracting the location and identity of individual mRNA molecules from raw images and building cell x gene expression matrices annotated with spatial coordinates of each cell

* Support of a next-generation chunked file format amenable to distributed processing of images from multiple
  fields of view, to enable fast, parallel processing of large volumes of image data.

* Outputs spatially-annotated gene expression matrices that interface directly with popular single-cell RNAseq analysis packages for downstream analysis and integration with scRNA-seq datasets

This document describes the work we have prioritized for the coming months.
If you have questions or feedback about this roadmap, please submit an issue on GitHub.
If there is work not on our roadmap that is important to your project, please consider :ref:`contributing`.

*Please note: this roadmap is subject to change.*

Last updated: November 25, 2019

What we are building now
----------------------------

Image segmentation improvements `#1321 <https://github.com/spacetx/starfish/issues/1321>`_
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
(see `scikit-image/scikit-image PR #3544 <https://github.com/scikit-image/scikit-image/pull/3544>`_),
we plan to support this in starfish.

Expanded support for image loading `#1603 <https://github.com/spacetx/starfish/issues/1603>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reformatting large image datasets is a hassle. To make it easier for researchers to use starfish without converting to
the SpaceTx image data format, we plan to expose and document key components of our TileFetcher, which will allow users
to write loaders for their own file formats.

Comprehensive support for 3D image processing `#53 <https://github.com/spacetx/starfish/issues/53>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While most of starfish supports 3D data processing, there are a few places where 3D support is lacking.
While we won’t be developing new algorithms to support 3D, we’re planning to make sure that starfish supports
3D wherever the underlying algorithms do.

Access to datasets `#1677 <https://github.com/spacetx/starfish/issues/1677>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starfish includes a module that gives users programmatic access to select public datasets.
We plan to expand the datasets that are available, improve documentation of this resource,
and generally improve access to these data.

Ensuring reliability and performance `#1309 <https://github.com/spacetx/starfish/issues/1309>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are a number of areas where we seek to improve the reliability and performance of starfish,
especially for users relying on starfish in production environments. These include improving
logging and versioning file formats, and improvements to performance and memory use.

Refreshed Documentation `#858 <https://github.com/spacetx/starfish/issues/858>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We want starfish’s documentation to be a resource not only for users of starfish, but for researchers
who are interested in adopting these assays, even if they haven’t generated their own data yet.
To do this, we are going to focus on building out tutorials and how-tos that not only help new users
learn get setup with starfish more quickly, but demonstrate how to gain qualitative insight into
experiment quality and understand the implications of image processing decisions on downstream data quality.

What's on the horizon
-------------------------

Support for proteomics assays `#1674 <https://github.com/spacetx/starfish/issues/1674>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Proteomics assays like CODEX, CyCIF, MIBI, IMC, and others follow similar image processing patterns
to RNA-based assays, but have some unique requirements that we expect to require additional algorithms
and components in starfish. We think adding support for these assays and hybrid transcriptomics/proteomics
assays would be valuable and would welcome community contributions.

Support for segmentation-free cell assignment `#1675 <https://github.com/spacetx/starfish/issues/1675>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are a number of compelling approaches to building single-cell gene expression matrices without
performing a classic image-based segmentation to define cell boundaries. These methods, including
`Baysor <https://github.com/hms-dbmi/Baysor>`_, `ssam <https://github.com/eilslabs/ssam>`_, and `cell_call <https://github.com/acycliq/cell_call>`_, create gene expression matrices directly from spots. We are very interested
in supporting these methods in starfish and would welcome community contributions.
