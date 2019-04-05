.. _example_workflow:

Example Data Processing Workflow
================================

Using the API
-------------

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

.. literalinclude:: ../../_static/data_processing_examples/iss_pipeline.py

Using the CLI
-------------

Starfish has a fully-featured CLI that mimics the API such that one does not need to write python
code to use starfish. Running the commands below will re-produce the notebook_ results using
starfish's CLI:

.. _notebook: ../../_static/data_processing_examples/iss_pipeline.py

.. literalinclude:: ../../_static/data_processing_examples/iss_cli.sh
