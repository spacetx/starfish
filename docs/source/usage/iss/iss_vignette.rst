In-Situ Sequencing (ISS)
========================

This in-situ sequencing (ISS) experiment has 16 fields of view that measure 4 channels on 4 separate
imaging rounds.

Downloading Data
----------------

Like all starfish data, this experiment is hosted on amazon web services. Once formatted,
experiments can be downloaded on-demand into starfish. However, downloading the raw data requires
that you access the data from an aws account.

For the purposes of this vignette, we will format only two of the 16 fields of view. To download the
data, you can run the following commands:

.. code-block:: bash

    mkdir -p iss/raw
    aws s3 cp s3://czi.starfish.data.public/browse/raw/20180820/iss_breast/ iss/raw/ \
        --recursive \
        --exclude "*" \
        --include "slideA_1_*" \
        --include "slideA_2_*" \
        --no-sign-request
    ls iss/raw

This command should download 44 images:

- 2 fields of view
- 2 overview images: "dots" used to register, and DAPI, to localize nuclei
- 4 rounds, each containing:
- 4 channels (Cy 3 5, Cy 3, Cy 5, and FITC)
- DAPI nuclear stain


Formatting single-plane TIFF files in sptx-format
-------------------------------------------------

We provide some tools to take a directory of files like the one just downloaded and translate it
into starfish-formatted files. These objects are ``TileFetcher`` and ``FetchedTile``. In brief,
TileFetcher provides an interface to get the appropriate tile from a directory for a given
set of sptx-format metadata, like a specific z-plane, imaging round, and channel by decoding
the file naming conventions . ``FetchedTile`` exposes methods to extract data specific to each
tile to fill out the remainder of the metadata, such as the tile's shape and data.

These are the abstract classes that must be subclassed for each set of naming conventions:

.. literalinclude:: /../../starfish/experiment/builder/providers.py
    :pyobject: FetchedTile

.. literalinclude:: /../../starfish/experiment/builder/providers.py
    :pyobject: TileFetcher

To create a formatter object for in-situ sequencing, we subclass the ``TileFetcher`` and
``FetchedTile`` by extending them with information about the experiment. When formatting
single-plane TIFF files, we expect that all metadata needed to construct the ``FieldOfView``
is embedded in the file names.

For the ISS experiment, the file names are structured as follows

.. code-block:: bash

    slideA_1_1st_Cy3 5.TIF

This corresponds to

.. code-block:: bash

    (experiment_name)_(field_of_view_number)_(imaging_round)_(channel_name).TIF

So, to construct a ``sptx-format`` ``FieldOfView`` we must adjust the basic TileFetcher object so
that it knows about the file name syntax.

That means implementing methods that return the shape, format, and an open file handle for a tile.
Here, we implement those methods, and add a cropping method as well, to mimic the way that ISS data
was processed when it was published.

.. literalinclude:: /../../data_formatting_examples/format_iss_breast_data.py
    :pyobject: IssCroppedBreastTile

This object, combined with a ``TileFetcher``, contains all the information that ``starfish`` needs
to parse a directory of files and create ``sptx-format`` compliant objects. Here, two tile fetchers
are needed. One parses the primary images, and another the auxiliary nuclei images that will be
used to seed the basin for segmentation.

.. literalinclude:: /../../data_formatting_examples/format_iss_breast_data.py
    :pyobject: ISSCroppedBreastPrimaryTileFetcher

.. literalinclude:: /../../data_formatting_examples/format_iss_breast_data.py
    :pyobject: ISSCroppedBreastAuxTileFetcher

Creating a Build Script
-----------------------

Next, we combine these objects with some information we already had about the experiments. On the
outset we stated that an ISS experiment has 4 imaging rounds and 4 channels, but only 1 z-plane.
These data fill out the ``primary_image_dimensions`` of the ``TileSet``. In addition, it was stated
that ISS has a single ``dots`` and ``nuclei`` image. In ``starfish``, auxiliary images are also
stored as ``TileSet`` objects even though often, as here, they have only 1 channel, round, and
z-plane.

We create a dictionary to hold each piece of information, and pass that to
``write_experiment_json``, a generic tool that accepts the objects we've aggregated above and
constructs TileSet objects:

.. literalinclude:: /../../data_formatting_examples/format_iss_breast_data.py
    :pyobject: format_data

Finally, we can run the script. We've packaged it up as an example in ``starfish``. It takes as
arguments the input directory (containing raw images), output directory (which will contain
formatted data) and the number of fields of view to extract from the raw directory.

.. code-block:: bash

    mkdir iss/formatted
    python3 data_formatting_examples/format_iss_breast_data.py iss/raw/ iss/formatted 3
    ls iss/formatted/*.json

Constructing a Pipeline
-----------------------

Now the images can be loaded and processed with ``starfish``!

.. code-block:: python

    >>> import starfish
    >>> exp = starfish.Experiment("iss/formatted/experiment.json")
    >>> exp

The ISS publication clearly describes how the data should be analyzed. The tiles within each stack
are registered using the provided dots images, and filtered with a ``WhiteTopHat`` filter to
increase rollony contrast against background and to remove large blobs that correspond to
auto-fluorescence artifacts.

The filtered images are then subjected to a `GaussianSpotDetector` that uses a
Laplacian-of-Gaussians approach to detect bright spots against a darker local background. Spot
"traces" are constructed by measuring the brightness of each spot across the channels and imaging
rounds. These spot traces are then decoded to determine which gene they represent using the
codebook.

Simultaneously, the data are segmented using an image of a DAPI nuclear stain to seed watershed
basins.

Finally, decoded spots are assigned to cells, producing a table wherein each spot is annotated with
spatial coordinates, gene, cell, and a quality score that measures how close to the predicted
barcode a given trace was.

``starfish`` exposes methods to accomplish each of the above tasks, which can be strung together
to create a pipeline that can be run either on the API, or using starfish's CLI. This vignette will
demonstrate the API.

The above steps can be recapitulated using starfish as follows:

.. literalinclude:: iss_pipeline.py

Visualizing Outputs
-------------------

Starfish loops into a variety of visualization ecosystems, and has some of its own plotting methods
in ``starfish.plot``, but it can also output spot data for visualization with high-performance WebGL
based tools like MERmaid.

To dump data for use with MERmaid, simply call ``decoded_intensities.save_mermaid()`` and then
follow the installation instructions for MERmaid.
