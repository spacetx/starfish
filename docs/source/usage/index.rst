.. _usage:

Starfish Usage
==============

The purpose of this short document is to concisely specify how data
needs to be pre-processed and formatted to be analyzed by *starfish*
while the package is in pre-alpha development. This document primarily
serves to condense and make minor clarifications to the `original
proposal <https://docs.google.com/document/d/12V-BQF-wh1GDBXwUH2hQrPhw6n0oQqENPK-YHGv59w4/edit#heading=h.r14mr5jpjh7r>`__
that was circulated earlier this year.

We remain committed, in 2019, to expand the capabilities of starfish
such that it can serve as a general purpose tool for the processing of
Image-based transcriptomics workflows, and are very thankful for the
feedback we've received on what will be required to accomplish that. We
hope that with those additions, *starfish* can reduce development burden
and speed processing and iteration. However, the full set of features
needed to accomplish this across assays are not yet complete. As a
result, some pre-processing must be applied to data before it can be
used in *starfish*.

Starfish Usage Checklist
------------------------

1. Image data

   a. In SpaceTx Format and validates with starfish validate,

   b. Corrected for optical aberration,

      i.   *starfish* contains some experimental filters to enable
           linear unmixing and flat field correction. These are not yet
           fully tested, so use these at your own risk. More complex
           transforms must be pre-applied.

   c. Registered

      i.   If the data are from a multiplexed assay, then starfish
           expects labs to send us pre-registered images, such that each
           image tile in a FOV, across rounds, covers data from the same
           coordinate range.

      ii.  If the data are from a non-multiplexed assay, then we either
           expect labs to send us pre-registered images as above, or for
           labs to enter the post-registered coordinates of each tile in
           the metadata. In the latter case, we expect labs to handle
           registration artifact post-processing.

      iii. If data require *only* translation registration, and authors
           provide a reference image against which all rounds can be
           registered, starfish provides some rudimentary tooling to
           support this registration.

2. A codebook in SpaceTx format

3. DAPI overview image of entire sample (if using *starfish*\ ’s
   segmentation tools)

Data Files and Formats
----------------------

Data and Auxiliary Images
~~~~~~~~~~~~~~~~~~~~~~~~~

Each dataset should be formatted in `SpaceTx
Format <https://spacetx-starfish.readthedocs.io/en/latest/sptx-format/specification.html>`__.
Briefly, SpaceTx format specifies an index, written in json, that
specifies metadata about a set of image tiles. It is built
hierarchically, where each Experiment corresponds to a microscope slide,
which contains multiple fields of view, which correspond to the
individual imaging locations and be the size of the camera's sensor, in
pixels. Common sizes that we have received include 1024 x 1024 and 2048
x 2048 pixel images. Each Field of View is built from a series of 2d
images, taken from each of the channels, rounds, and focal planes that
are captured in that physical location over the experiment.

Auxiliary images that are captured for reasons other than identifying
the localization of mRNA transcripts, such as images of anchor probes,
nissl stains, dapi stains, or fiduciary beads, should be stored as
Auxiliary images that are associated with each field of view.

Constructed Experiment objects can be validated with starfish validate.
Instructions for carrying out this validation can be found
`here <https://spacetx-starfish.readthedocs.io/en/latest/usage/validation/>`__.
If any issues arise during this process, please open an issue and let us
know.

*starfish* provides tooling to create Experiment objects automatically
in cases where images are stored in a local directory structure, and the
directories or file names contain all information about the fluorescence
channel, imaging round, z-plane, physical coordinates, and field of view
for data and auxiliary images. *starfish* provides examples of how
naming conventions can be used to extract data for
`MERFISH <https://github.com/spacetx/starfish/blob/master/examples/get_merfish_u20s_data.py#L1>`__,
`ISS <https://github.com/spacetx/starfish/blob/master/examples/get_iss_breast_data.py#L1>`__,
or
`osmFISH <https://github.com/spacetx/starfish/blob/master/examples/format_osmfish.py#L1>`__
datasets.

Codebook
~~~~~~~~

Each dataset should be accompanied by a
`codebook <https://spacetx-starfish.readthedocs.io/en/latest/sptx-format/index.html#codebook>`__
in SpaceTx format with the filename codebook.json. The codebook is also
stored in JSON, and specifies, for each target mRNA, the expected
fluorescence value for each round and channel of the experiment. We've
provided some toy examples that demonstrate what targets might look like
in both sequential smFISH and multiplex experiments.

Data Pre-processing
-------------------

We have prioritized the implementation of solutions that generalize
across studies, such as image filtering, spot calling, decoding,
segmentation, and quantification.

An early learning from *starfish* has been that each assay and
microscope have different quirks. The subtle changes in position of the
microscope stage, and the optical properties of the microscopes and
cameras, require diverse solutions to register images and correct
optical aberrations. As a result, we have not yet been able to complete
the implementation of general solutions to correct for these issues, and
request users pre-process data as described below.

Data should be pre-registered and if the assay is code-based, the transformation should be applied
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*starfish* currently requires users pre-register their data because the
requirements and associated algorithms are highly variable across
methods and have strong dependencies on signal acquisition properties.
Specifically, some methods absolutely require sub-pixel registration of
all transcripts across rounds, while others can maintain a high degree
of accuracy in transcriptomic profiles with registration errors up to
several pixels. However, in all cases the output is the same: coordinate
information for each image tile in the Field of View that is of
sufficient precision to allow subsequent processing. Since this output
is common to all methods, we are asking that the (x, y, z) location of
each image tile provided in the data manifest to be the
*post-registration location*. If a spaceTx method requires image
transformations into some registered coordinates (e.g. for spectral
unmixing before spotfinding, or decoding spots or pixels across rounds),
data contributors must provide *transformed* image data.

This means that for single-molecule FISH, for which each imaging round
and channel are independent, data must be pre-registered, where
registration is defined as finding the correct spatial localization of a
tissue tile with respect to all other adjacent tiles. For
multiplexed/coded assays like MERFISH, ISS, BaristaSeq, StarMAP,
SeqFISH, and MExFISH, this means that data must be registered *and
resliced by applying the learned transformation*, and the
post-registration physical image coordinates must be identical for all
tiles in the same field of view.

Data should be corrected for optical aberrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image-based transcriptomics methods may require correction of residual
chromatic or other optical aberrations such as pincushioning or
nonuniform excitation intensities. These corrections, which are assumed
to be correctable before any image-based transcriptomics pipeline
processing, should be applied by the data contributor to image data
before processing with the pipeline.

The pixel intensities of hybridization signals, background and
autofluorescence may vary across hybridization rounds (H), color
channels (C) and even across species, as well as between different
imaging methods. The primary mechanism to deal with variable dynamic
range is correctly setting pipeline recipe parameters for the Image
Filtering pipeline component, but in the event that those results are
inadequate for downstream processing, data contributors may pre-scale
image data. Additional modules to handle specific image intensity
scaling problems are welcome as part of the Image Filtering pipeline
component.

No other processing should be applied to the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only registration and optical aberrations should be corrected prior to
analysis with *starfish*. Background subtraction, for example, is
handled by Starfish. If there is confusion about what types of image
pre-processing should be applied, please open an issue.


Getting started with the CLI
============================

The simplest way to get started with starfish for most users will be to try out the
command-line interface (CLI). After following the :ref:`installation <installation>`
instructions, a ``starfish`` command will be available. Running ``starfish --help``
will print out the subcommands that are available.

.. program-output:: env MPLBACKEND=Agg starfish --help

.. toctree::
   :maxdepth: 3
   :caption: CLI:

.. toctree::
   fov-builder/fov-builder.rst

.. toctree::
   validation/index.rst

.. toctree::
   configuration/index.rst

Vignettes
=========

This section provides several end-to-end usage vignettes for applying starfish to image-based
transcriptomics data. The first vignette provides an example of using starfish to format a small,
16 field of view experiment leveraging the in-situ sequencing (ISS) approach.

.. toctree::
   :maxdepth: 2
   :caption: Vignettes:

.. toctree::
   iss/iss_vignette.rst

.. toctree::
   iss/iss_cli_vignette.rst
