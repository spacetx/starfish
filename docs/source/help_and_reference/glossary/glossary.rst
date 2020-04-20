Here, we clearly define the relevant terms needed to understand the spaceTx pipeline specification

Glossary
--------

.. glossary::

    Channel
        An imaging mode that captures a continuous-valued feature from a field of view. Examples of channels include the read-out from a fluorescent dye, such as Cy3, or a the abundance of an isotope captured from a mass spectrometer.

    Imaging Round
        Several image-based transcriptomics and proteomics approaches will image the same tissue multiple times. Each time the tissue is imaged is a discrete imaging round.

    Target
        A feature that is the target of quantification by an image-based assays. Common targets include mRNA transcripts or proteins.

    IntensityTable
        An intensity Table contains the features identified in an ImageStack. It can be thought of as an array whose entries are the intensities of each feature across the imaging rounds and channels of a field of view. Starfish exposes several processing tools to decode the features of the table, estimate their qualities, and assign features to cells.

    DecodedIntensityTable
        A representation of a decoded intensity table. Contains the features identified in an ImageStack as well as their associated target values.

    Codeword
        A codeword maps expected intensities across multiple image tiles within a field of view to the target that is encoded by the codeword.

    Codebook
        A codebook contains all the codewords needed by an experiment to decode an IntensityTable. It also contains a mapping of channels to the integer indices that are used by starfish to represent them internally.

    Pipeline
        A sequence of data processing steps to process the inputs into the desired outputs.

    Pipeline Component
        A single data processing step in the pipeline, as defined by its input and output file formats, e.g., the spot-detection component takes as input an image and outputs a table of spot locations, shapes, and intensities.

    Pipeline Component Algorithm
        A specific algorithm type that adheres to the specified input and output file formats required by the component it belongs to. For example, a spot-detection component algorithm can be realized as a Gaussian blob detector or a connected components labeller. Both find spots and accept the same inputs and produce the same outputs, hence belong to the same component. However, the underlying properties of the algorithms (and parameterizations) may be quite different.

    Pipeline Specification
        A document describing the pipeline in detail, including an ordered list of all the pipeline components, and expected input/output file formats at each step of computation.

    Pipeline Implementation
        Actual code for the pipeline. This code will be packaged as a well-documented Python library and corresponding command line tool for use by consortium members to facilitate easy sharing and comparison of results across labs/methods.

    Manifest
        The data manifest is a file that includes the locations of all fields of view for either primary or auxiliary images.

    Field of View (FOV)
        A collection of Image Tiles corresponding to a specific volume or plane of the sample, under which the signal for all channels and all imaging rounds were acquired. All tiles within this FOV are the same size, but the manifest allows for different spatial coordinates for different imaging rounds or channels (to accommodate slight movement between rounds, for example).
        In microscopy, a field of view corresponds to the camera sensor mapped to the sample plane, and many such fields of view are expected to be taken per tissue slice.

    Region of Interest (ROI)
        Areas of an image identified for a particular purpose, such as to define the boundaries of a cell.

    Image Tile
        A single plane, single channel, single round 2D image. In the manifest, each tile has information about its (X,Y,Z) coordinates in space, and information about which imaging round (R) and/or fluorescence channel (C) it was acquired under.

    ImageSlice
        The image volume corresponding to a single round and single channel of the Field of View.

    Coordinates (Tile)
        Coordinates refer to the physical location of a Tile with respect to some independent reference.  If a pair of values are provided, it corresponds to the physical coordinates of the edges.  If a single value is provided, it corresponds to the center of the tile.  For x and y, two values are required.  For z, both a single value and a pair of values are valid.

    Axes (Tile)
        Each pixel is located along five axes, which are round, channel, z-plane, y, and x.

    Labels (Tile)
        Labels are the valid set of values for each axis.

    Index (Tile)
        An index indicates the label or range of labels for a given axis.  These should be a whole number (non-negative integers) or a python contiguous slice representing a range.

    Selectors (Tile)
        A mapping of axes (round, channel, and z-plane) to their respective index.  These are expressed as a mapping from Axis to index.

    Primary Images
        The primary image data for an experiment. Primary images contain information on imaging targets. primary images build fields of view that usually contain multiple channels and may contain multiple imaging rounds. Primary images can be decoded to identify the abundance of transcript or protein targets.

    Auxiliary Images
        Any user-submitted additional images for analysis beyond the primary images. These images may be of lower dimension than the primary images (e.g., single channel images), but should span the same spatial extent as the primary images acquired under the same FOV. Auxiliary images are used to aid the image processing of the primary images.

        Examples of such data may include:

        Nuclei (DAPI or similar nuclear stain): this required image shows cell nuclei and is crucial for cell segmentation further on down the pipeline.

        Dots: an image containing the locations of imaging features across a field of view.

        Other stains or labels: these optional (but recommended) image(s), including but not limited to antibody stains, may capture additional information about cell boundaries or subcellular structure that will be useful for cell segmentation and/or additional spatial analyses.

    Registration
        Refers to the process of aligning multiple images of the same spatial location, mostcommonly across multiple rounds of imaging within a FOV.

    Stitching
        The process of combining images from multiple fields of view into a larger image thatspans the extent of the sample.

    Feature
        The value of a spot (aggregated across all pixel values circumscribed by that spot) or the value of a single pixel.

    Feature (Spot, Pixel) Trace
        Feature intensity values across all imaging rounds and/or color channels. These map to codewords in a codebook.

    Decoding
        Matching putative barcodes to codewords in a codebook to read out the corresponding target believed to be associated with that barcode.

    Rolony
        A rolling-circle amplified "colony", or rolony, is an amplicon produced by image-based transcriptomics assays that use circular probes to increase signal.