.. _introduction:

Introduction
============

*starfish* is a Python library which lets you build scalable pipelines for processing image-based
transcriptomics data. We are in active development.


Image-based transcriptomics
~~~~~~~~~~~~~~~~~~~~~~~~~~~
In-situ image based transcriptomics refers to a collection of technologies for
characterizing spatial patterns of gene expression within and across cells with
high detection efficiency and throughput. These methods are likely to transform
the single-cell genomics community by complementing existing single cell
sequencing-based techniques.

With *starfish*, we are engineering a data ingestion and analysis pipeline that
fosters usability, robustness, and reproducibility within and across labs.

Several image-based transcriptomics methods have been reviewed in the 2015
Nature Reviews Genetics article Spatially Resolved Transcriptomics and Beyond.
A recent 2017 review in Current Opinion in Biotechnology focuses on why these
methods will be important for advancing our understanding of tissue organization
and design — Spatial transcriptomics: paving the way for tissue-level systems
biology.

We consider methods based specifically on sub-cellular resolution fluorescence
microscopy. In particular, we consider multiplexed single molecule fluorescent
in situ hybridization (smFISH) and multiplexed in-situ sequencing (ISS).
Conceptually, both methods work by attaching fluorescently labeled probes onto
target sequences that specify genes on mRNA molecules. These probes are
subsequently visualized as a sequence of diffraction limited spots by
fluorescence microscopy across multiple rounds of hybridization; consider this
sequence a barcode. The target gene is then determined from the barcode with a
decoding algorithm.

It’s worth noting that spatial transcriptomics — a promising new method that
uses a special glass slide to capture spatially-resolved mRNA from a tissue
section for in-vitro sequencing — is not considered here. This method is
considerably simpler than the methods described above; however, it results in
relatively coarse spatial resolution (tissue level vs. sub-cellular) and lower
transcript detection efficiency. Its data analysis challenges are also closer
to those used for sequencing than for the imaging methods considered here.

*starfish* has been designed to support most of the major image-based
transcriptomics assays currently in use by the groups in the :ref:`SpaceTx consortium `spacetx``.


Pipelines
~~~~~~~~~

The diagram below describes the core pipeline components and
associated file manifests that this package plans to standardize and implement.

.. _document: https://docs.google.com/document/d/1IHIngoMKr-Tnft2xOI3Q-5rL3GSX2E3PnJrpsOX5ZWs/edit?usp=sharing

.. image:: /_static/design/pipeline-diagram.png
    :alt: pipeline diagram

Corrections
-----------

Spatial
^^^^^^^

There may be sample movement and distortion across imaging rounds, which limits
the ability to detect the same mRNA molecule in the same location across rounds.
To correct for this, one typically learns an affine transformation (translation,
rotation, scaling) to align all images to a specified reference within each FOV.

*starfish* currently supports only a basic implementation of this step.

Chromatic
^^^^^^^^^
There may be chromatic aberration across color channels (that is potential
spatially dependent) based on the microscope, laser lines, and signal
amplification approach used by the experimenter. This aberration can be
corrected by imaging multi-spectral fiduciary beads (once per experiment), then
learning a correction matrix which can be subsequently applied across channels
for all fields of view. We propose to assume that data providers have already
applied corrections if required.

*starfish* does not currently support this step.

Signal Intensity
^^^^^^^^^^^^^^^^
Signal intensities may vary in dynamic range across hybridization rounds (H) and
color channels (C). We propose two methods to correct for this:

*starfish* does not currently support this step.

Filtering
---------

The raw data should be filtered sequentially for three reasons:

To de-convolve the point spread function (PSF) of the imaging system. This will
be done with the Richardson-Lucy deconvolution algorithm if the PSF is known and
submitted a-priori. Else, this step will not be performed.

- To remove background signal. This will be done with a Gaussian high pass filter.
- To enhance spot detection. This will be done with a white top-hat or Gaussian band-pass filter.

All filters will be applied across all dimensions in all FOVs. The filters all
have parameters that need to be set by the experimenter through visual inspection
and these parameters typically depend on the experiment, microscope, and desired
decoding method. In principle, we can provide sensible defaults, at a minimum
based on derived image statistics, and ideally automatically inferred from
experimental properties.

Spot Detection
--------------

The mRNA molecules in the filtered images are represented as diffraction limited
spots, which form sets of connected or nearly-connected pixels. These spots need
to be detected to obtain a position in space (x,y,z) and possibly a radius (r).
Signals within each spot can then be integrated to yield an H dimensional vector
(barcode) that represents either the mean or maximum intensity of the spot across
each hybridization round.

We will support three options for spot detection:
- Multi-scale Laplacian of Gaussian (LoG) filtering
- Connected components labeling.
- Consider each ‘spot’ as a single pixel. Perform spot detection using connected
components labelling after each pixel is decoded into a gene.

We are still actively exploring whether spot-detection should be performed before
or after decoding. Performing it before provides a cleaner signal per spot for
subsequent decoding, and reduces the volume of data. Performing it after
potentially allows non-linear signal refinement through decoding that in turn
may result in cleaner detection and improved signal-to-noise.

Segmentation
------------

Spots (representing RNA molecules) must be assigned to regions (representing
cells) to characterize expression patterns per cell (e.g. yield a gene x cell
table). In principle, assignment need not require explicit segmentation of
regions, but explicit regions are required for many other spatial analyses. We
propose to support these operations through the watershed segmentation
algorithm, which determines cell boundaries as polygons, followed by the
assignment of spots to cells using a point-in-polygon algorithm based on
ray-tracing.

The Watershed algorithm requires two inputs: a seed image (which can be derived
from the DAPI stain) and an input image. We will support two forms of input image:

- The user provides a membrane stain (preferred)
- We max project the stack across H and C, then estimate the spatial density of molecules by applying a Gaussian low pass filter.

The watershed algorithm has parameters that need to be set by the experimenter
through visual inspection and these parameters typically depend on the
experiment and microscope. As with filtering, in principle sensible defaults can
be obtained by learning a mapping from experimental details to analysis patterns.

Additionally, we are actively exploring algorithms that use manually labelled
region boundaries to train supervised machine learning models (e.g. convolutional
deep networks) to predict region boundaries on new unlabeled data.

Decoding
--------

The aim of decoding is to translate the encoder table into a decoder table that
has an explicit gene for each spot ID.

This decoder table, combined with the spot-cell table (joined on spot_id), and
coupled with the geoJSON files for visually depicting cells and spots, form a
complete dataset for many downstream analyses.

We are still actively exploring whether there is a unifying decoding algorithm
that works for all methods, whether we need to support user defined functions
(UDFs) that are specific to each assay type, or whether we can support a compact
set of algorithms that are more broadly method dependent.

Quality Control
---------------

We propose to generate, at a minimum, the following quality control metrics:

-  Spot size (measured in pixels)
- Spot intensity (low is bad)
- Region size (measured in pixels, small is bad)
- Number of genes per cell

Ideally, we will also be able to compare gene copy numbers with an alternative
assay like non-multiplexed, smFISH, scRNA-seq, or bulk sequencing. We will report
a correlation coefficient across assays as a guiding quality metric.

Furthermore, depending on the precise form of error correction (if any) we will
report metrics about the types and rates of errors that were detected and/or
corrected.
