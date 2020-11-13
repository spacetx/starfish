.. _user_guide:

User Guide
==========

Welcome to the user guide for building an image processing pipeline using starfish! This tutorial
will cover all the steps necessary for going from raw images to a single cell gene expression
matrix. If you only have a few minutes to try out starfish, check out the
:ref:`Quick Start<quick start>` to see a demonstration of how starfish works. starfish
requires a working knowledge of Python and fluorescent image analysis to create an analysis pipeline.
If you are ready to learn how to build your own image processing pipeline using starfish then read on!

This part of the tutorial goes into more detail about why each of the stages in the example are
needed, and provides some alternative approaches that can be used to build similar pipelines.

The core functions of starfish pipelines are the detection (and :term:`decoding<Decoding>`)
of spots, and the segmentation of cells. Each of the other approaches are designed to address
various characteristics of the imaging system, or the optical characteristics of the tissue
sample being measured, which might bias the resulting spot calling, decoding, or cell
segmentation decisions. Not all parts of image processing are always needed; some are dependent
on the specific characteristics of the tissues. In addition, not all components are always found
in the same order. *Starfish* is flexible enough to omit some pipeline stages or disorder them,
but the typical order might match the following. The links show how and when to use each
component of *starfish*, and the final section demonstrates putting together a "pipeline recipe"
and running it on an experiment.

.. _section_formatting_data:

Formatting Data
---------------

In order to load the experiment into a starfish pipeline the data must be in
:ref:`SpaceTx Format<sptx_format>`, which is a standardized format that utilizes json
files to organize single-plane tiffs for image-based spatial transcriptomics data. If the data
you want to process isn't already in SpaceTx Format, there are a few methods to convert
your data.

.. note::

    When converting data to SpaceTx Format is too costly, images can be loaded directly without
    formatting by :ref:`tilefetcher_loader`. This is a workaround and only recommended if
    reading and writing all the images is infeasible. The experiment JSON files like the codebook
    will still need to be created.

The primary method is to use :py:func:`.format_structured_dataset`, a conversion tool, on
data that is structured as 2D image tiles with specific filenames and a CSV
file containing the physical coordinates of each image tile. This method requires minimal Python
knowledge. You can manually organize your images, but for large datasets you will want to use a
script (e.g. Matlab) to move and rename files into the structured data format. The structured
data must be 2D image tiles in TIFF, PNG, or NPY file format.

Users who are familiar with Python and starfish also have the option of using
:py:class:`.TileFetcher` and :py:class:`.FetchedTile`, a set of user-defined interfaces the
experiment builder uses for obtaining the data corresponding to each tile location. Any data
format that can be read as a numpy array can be used.

Lastly, there is a 3rd party `spacetx writer`_ which writes SpaceTx-Format experiments using the
`Bio-Formats`_ converter. Bio-Formats can read a variety of input formats, so might be a
relatively simple approach for users familiar with those tools.

.. _spacetx writer: https://github.com/spacetx/spacetx-writer
.. _Bio-Formats: https://www.openmicroscopy.org/bio-formats/

After converting, you can use :ref:`starfish validate<cli_validate>` to ensure that the experiment
files meet the format specifications before loading.

Your first time applying these generalized tools to convert your data can be time-consuming. If
you just want to try starfish before investing the time to format your data, you can use one of the
:ref:`formatted example datasets <datasets>` included in the starfish library.

* Tutorial: :ref:`Formatting structured data<format_structured_data>`
* Tutorial: :ref:`Formatting with TileFetcher<format_tilefetcher>`

.. _section_loading_data:

Loading Data
------------

Once the data is in :ref:`SpaceTx Format<sptx_format>`, loading the whole experiment into starfish
is simple. The only options are for selecting which :term:`FOVs <Field of View (FOV)>` and
subsets to load into memory.

As mentioned in the previous section, it is also possible to
:ref:`directly load data <tilefetcher_loader>` that has not been formatted, although
there may be performance implications in doing so. This method is also more complicated.

* Tutorial: :ref:`Loading SpaceTx Formatted Data <loading_data>`
* Tutorial: :ref:`Loading Data Without Formatting <tilefetcher_loader>`

.. _section_manipulating_images:

Manipulating Images
-------------------

Sometimes it can be useful subset the images by, for example, excluding out-of-focus images or
cropping out edge effects. For sparse data, it can be useful to project the z-volume into a single
image, as this produces a much faster processing routine. Starfish supports the cropping and
projecting of :py:class:`.ImageStack`\s with the :py:meth:`.sel` and :py:meth:`.reduce` methods.

* Tutorial: :ref:`Cropping <tutorial_cropping>`
* Tutorial: :ref:`Projecting <tutorial_projection>`

.. _section_correcting_images:

Correcting Images
-----------------

These stages are typically specific to the microscope, camera, filters, chemistry, and any tissue
handling or microfluidices that are involved in capturing the images. These steps are typically
*independent* of the assay. *Starfish* enables the user to design a pipeline that matches their
imaging system and provides some basic image correction methods.

* Tutorial: :ref:`Illumination Correction <tutorial_illumination_correction>`
* Tutorial: :ref:`Image Registration <tutorial_image_registration>`

.. _section_improving_snr:

Enhancing Signal & Removing Background Noise
--------------------------------------------

These stages are usually specific to the sample being analyzed. For example, tissues often have
some level of autofluorescence which causes cellular compartments to have more background noise than
intracellular regions. This can confound spot finders, which look for local intensity differences.
These approaches ameliorate these problems.

* Tutorial: :ref:`Removing Autofluorescence <tutorial_removing_autoflourescence>`

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

Normalizing Intensity Distributions
"""""""""""""""""""""""""""""""""""

If you know a priori that image volumes acquired for every channel and/or every round should have
the same distribution of intensities then the intensity *distributions* of image volumes can be
normalized with :py:class:`.MatchHistograms`. Typically this means the number of spots and amount of
background autofluorescence in every image volume is approximately uniform across channels and/or
rounds.

* Tutorial: :ref:`Normalizing Intensity Distributions<tutorial_normalizing_intensity_distributions>`

Normalizing Intensity Values
""""""""""""""""""""""""""""

In most data sets the differences in gene expression leads to too much variation in number of
spots between channels and rounds. Normalizing intensity distributions would incorrectly skew the
intensities. Instead you can use :py:class:`.Clip`, :py:class:`.ClipPercentileToZero`, and
:py:class:`.ClipValueToZero` to normalize intensity *values* by clipping extreme values and
rescaling.

* Tutorial: :ref:`Normalizing Intensity Values <tutorial_normalizing_intensity_values>`

.. _section_finding_and_decoding:

Finding and Decoding Spots
--------------------------

Finding and decoding bright spots is the unique core functionality of starfish and is necessary in
every image-based transcriptomics processing pipeline. The inputs are all the images from a
:term:`FOV <Field of View (FOV)>` along with a :term:`codebook <Codebook>` that describes the
experimental
design. The output after decoding is a :term:`DecodedIntensityTable` that contains the
location, intensity values, and mapped :term:`target <Target>` of every detected
:term:`feature <Feature>`.

Every assay uses a set of rules that the :term:`codewords <Codeword>` in the codebook
must follow (e.g. each target has one hot channel in each round). These rules determine which
decoding methods in starfish should be used. See :ref:`section_which_decoding_approach` to
learn about different codebook designs and how to decode them.

There are two divergent decoding approaches, spot-based and pixel-based, used in the image-based
transcriptomics community when it comes to analyzing spots in images:

.. image:: /_static/design/decoding_flowchart.png
   :scale: 50 %
   :alt: Decoding Flowchart
   :align: center

Spot-Based Decoding
^^^^^^^^^^^^^^^^^^^

The spot-based approach finds spots in each image volume based on the brightness of regions
relative to their surroundings and then builds a :term:`spot trace<Feature (Spot, Pixel) Trace>`
using the appropriate :ref:`TraceBuildingStrategies<howto_tracebuildingstrategies>`. The spot
traces can then be mapped, or *decoded*, to codewords in the codebook using a
:py:class:`.DecodeSpotsAlgorithm`.

.. list-table::
   :widths: auto
   :header-rows: 1

   * - When to Use
     - How-To
   * - Images are amenable to spot
       detection methods
     - :ref:`howto_spotfindingresults`
   * - Data is from sequential methods
       like smFISH
     - :ref:`howto_simplelookupdecoder`
   * - Spots are sparse and may not be
       aligned across all rounds
     - :ref:`Use TraceBuildingStrategies.NEAREST_NEIGHBOR <howto_tracebuildingstrategies>`

* Tutorial: :ref:`Spot-Based Decoding with FindSpots and DecodeSpots <tutorial_spot_based_decoding>`

Pixel-Based Decoding
^^^^^^^^^^^^^^^^^^^^

The pixel-based approach first treats every pixel as a :term:`feature <Feature>` and constructs a
corresponding :term:`pixel trace<Feature (Spot, Pixel) Trace>` that is mapped to codewords.
Connected component analysis is then used to label connected pixels with the same codeword as an RNA
spot.

* Tutorial: :ref:`Pixel-Based Decoding with DetectPixels <tutorial_pixel_based_decoding>`

.. _section_which_decoding_approach:

What Decoding Pipeline Should I Use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are unsure which spot finding and decoding methods are compatible with your data here is a
handy table that summarizes the three major :term:`codebook <Codebook>` designs and what methods
can be used to decode each of them. If your codebook doesn't fall into any of these categories,
`make a feature request on github <https://github.com/spacetx/starfish/issues/new/choose>`_, we
would love to hear about unique codebook designs!

.. _tab-codebook-designs:

.. table::
   :class: "table-bordered"

   +-----------------+---------------------------+-------------------------+--------------------------+
   | Name            | Linearly Multiplexed      | One Hot Exponentially   | Exponentially Multiplexed|
   |                 |                           | Multiplexed             |                          |
   +=================+===========================+=========================+==========================+
   | Assays          | - sequential smFISH       | - In Situ Sequencing    | - MERFISH                |
   |                 | - RNAscope                | - seqFISH               | - DARTFISH               |
   |                 | - osmFISH                 | - FISSEQ                | - seqFISH+               |
   |                 |                           | - STARmap               |                          |
   |                 |                           | - BaristaSeq            |                          |
   +-----------------+---------------------------+-------------------------+--------------------------+
   | Example 7-round | |linear1|                 | |onehot1|               | |multiplex1|             |
   | Codebook        |                           |                         |                          |
   | Diagrams        | |linear2|                 | |onehot2|               | |multiplex2|             |
   +-----------------+---------------------------+-------------------------+--------------------------+
   | Description     | Codewords have only one   | Codewords are one hot   | Each codeword is a       |
   |                 | round and channel with    | in each round           | combination of signals   |
   |                 | signal                    |                         | over multiple rounds     |
   +-----------------+---------------------------+-------------------------+--------------------------+
   | Reference Image | No                        | Yes                     | Yes                      |
   | Needed?         |                           |                         |                          |
   +-----------------+---------------------------+-------------------------+--------------------------+
   | starfish        | - SimpleLookup            | - Exact_Match or        | - Pixel-based            |
   | Pipeline        | - Sequential +            |   Nearest_Neighbor      | - Exact_Match +          |
   | Options         |   PerRoundMaxChannel      | - PerRoundMaxChannel or |   MetricDistance         |
   |                 |                           |   MetricDistance        | - Nearest_Neighbor +     |
   |                 |                           |                         |   MetricDistance         |
   +-----------------+---------------------------+-------------------------+--------------------------+

.. |linear1| image:: /_static/design/linear_codebook_1.png
   :scale: 10%
   :align: middle
.. |linear2| image:: /_static/design/linear_codebook_2.png
   :scale: 10%
   :align: middle
.. |onehot1| image:: /_static/design/onehot_codebook_1.png
   :scale: 10%
   :align: middle
.. |onehot2| image:: /_static/design/onehot_codebook_2.png
   :scale: 10%
   :align: middle
.. |multiplex1| image:: /_static/design/multiplex_codebook_1.png
   :scale: 10%
   :align: middle
.. |multiplex2| image:: /_static/design/multiplex_codebook_2.png
   :scale: 10%
   :align: middle

.. _section_segmenting_cells:

Segmenting Cells
----------------

Unlike single-cell RNA sequencing, image-based transcriptomics methods do not physically separate
cells before acquiring RNA information. Therefore, in order to characterize cells, the RNA must be
assigned into single cells by partitioning the image volume. Accurate unsupervised cell-segmentation
is an `open problem <https://www.kaggle.com/c/data-science-bowl-2018>`_ for all biomedical imaging
disciplines ranging from digital pathology to neuroscience.

The challenge of segmenting cells depends on the structural complexity of the sample and quality
of images available. For example, a sparse cell mono-layer with a strong cytosol stain would be
trivial to segment but a dense heterogeneous population of cells in 3D tissue with only a DAPI stain
can be impossible to segment perfectly. On the experimental side, selecting good cell stains and
acquiring images with low background will make segmenting a more tractable task.

There are many approaches for segmenting cells from image-based transcriptomics assays. Below are
a few methods that are implemented or integrated with starfish to output a
:py:class:`.BinaryMaskCollection`, which represents a collection of labeled objects. If you do not
know which segmentation method to use, a safe bet is to start with thresholding and watershed. On
the other hand, if you can afford to manually define :term:`ROI <Region of Interest (ROI)>` masks
there is no better way to guarantee accurate segmentation.

.. note::
    While there is no "ground truth" for cell segmentation, the closest approximation is manual
    segmentation by an expert in the tissue of interest.

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
.html>`_. Watershed is a classic image processing algorithm for separating objects in images and
can be applied to all types of images. Pairing it with a distance transform is particularly
useful for segmenting convex shapes like cells.

A segmentation pipeline that consists of thresholding, connected component analysis, and watershed
is simple and fast to implement but its accuracy is highly dependent on image quality.
The signal-to-noise ratio of the cell stain must be high enough for minimal errors after
thresholding and binary operations. And the nuclei or cell shapes must be convex to meet the
assumptions of the distance transform or else it will over-segment. Starfish includes the basic
functions to build a watershed segmentation pipeline and a predefined :py:class:`.Watershed`
segmentation class that uses the :term:`primary images<Primary Images>` as the cell stain.

* Tutorial: :ref:`Ways to segment by thresholding and watershed<tutorial_watershed_segmentation>`

Manually Defining Cells
^^^^^^^^^^^^^^^^^^^^^^^

The most accurate but time-consuming approach is to manually segment images using a tool such as
`ROI manager <https://imagej.net/docs/guide/146-30.html#fig:The-ROI-Manager>`_ in FIJI (ImageJ). It
is a straightforward process that starfish supports by importing
:term:`ROI <Region of Interest (ROI)>` sets stored in ZIP archives to be imported as a
:py:class:`.BinaryMaskCollection`. These masks can then be integrated into the pipeline for
visualization and assigning spots to cells.

* Tutorial: :ref:`Loading ImageJ ROI set<tutorial_manual_segmentation>`

Machine-Learning Methods
^^^^^^^^^^^^^^^^^^^^^^^^

Besides the two classic cell segmentation approaches mentioned above, there are machine-learning
methods that aim to replicate the accuracy of manual cell segmentation while reducing the labor
required. Machine-learning algorithms for segmentation are continually improving but there is no
perfect solution for all image types yet. These methods require training data (e.g. stained
images with manually defined labels) to train a model to predict cell or nuclei locations in test
data. There are `exceptions that don't require training on your specific data <http://www.cellpose
.org/>`_ but generally training the model is something to consider when evaluating how much time
each segmentation approach will require.

Starfish currently has built-in functionality to support `ilastik <https://www.ilastik.org/>`_, a
segmentation toolkit that leverages machine-learning. Ilastik has a Pixel Classification
workflow that performs semantic segmentation of the image, returning probability maps for each
label such as cells and background. To transform the images of pixel probabilities to binary
masks, you can use the same thresholding and watershed methods in starfish that are used for
segmenting images of stained cells.

* Tutorial: :ref:`Using ilastik in starfish<tutorial_ilastik_segmentation>`

.. _section_assigning_spots:

Assigning Spots to Cells
------------------------

After segmenting images to find cell boundaries, RNA spots in the :py:class:`.DecodedIntensityTable`
can be assigned to cells and then the table can be reorganized to create a single cell gene
:py:class:`.ExpressionMatrix`. These matrices are the data structure most often generated and used
by single-cell RNAseq analysis packages (e.g. `scanpy <https://icb-scanpy.readthedocs-hosted
.com/en/stable/>`_) to cluster and classify cell types. Compared to single-cell RNAseq, image-based
transcriptomic methods provide additional information about the cell, such as its location, size,
and morphology. The :py:class:`.ExpressionMatrix` holds both the 2-Dimensional matrix and cell
metadata produced by these image-based methods. This data is what links the histological context of
single cells to their transcriptomes.

In a starfish pipeline, the first step to creating a gene expression matrix is assigning spots,
aka :term:`features <Feature>`, to cells defined in a :py:class:`.BinaryMaskCollection` as cell
masks. This is done by using :py:class:`.Label` to label features with ``cell_id``\s. Currently,
:py:class:`.Label` assumes every cell mask created by
:ref:`cell segmentation<section_segmenting_cells>` encompasses a whole cell. RNA spots
with spatial coordinates that are within a cell mask are assigned to that cell and spots that do
not fall within any cell mask are not assigned a ``cell_id``. Therefore, the accuracy and
percent yield of assigned spots is largely dependent on the quality and completeness of cell
segmentation.

For data without well segmented cells, such as when no cell stain images are available, there is
potential for more sophisticated methods to assign spots to cells. For example, there are a
number of segmentation-free approaches for grouping spots into cells that starfish would like to
support in the `future <https://github.com/spacetx/starfish/issues/1675>`_.

* Tutorial: :ref:`tutorial_assigning_spots`


.. _section_working_with_starfish_outputs:

Working with starfish outputs
-----------------------------

Once you've processed your data with starfish, you are ready to load the output
files into tools like Seurat and ScanPy for further analysis. Starfish lets you
save expression matrices and segmentation masks in a variety of data formats.

* Tutorial: :ref:`working_with_starfish_outputs`

.. _section_processing_at_scale:

Processing at Scale with AWS
----------------------------

When you are ready to scale up your analysis to a full experiment or multiple
experiments, starfish can be deployed on the cloud or an an institutional
high performance computing cluster for efficient analysis of large datasets.
Implementation details will vary based on the compute resources at your disposal,
but below we demonstrate how you can analysis a full dataset on AWS.

* Tutorial: :ref:`processing_at_scale`

.. _section_further_reading:

Further Reading
---------------

Additional resources are available in :ref:`help and reference`.



.. toctree::
   :hidden:

   working_with_starfish_outputs/index
   processing_at_scale/index
