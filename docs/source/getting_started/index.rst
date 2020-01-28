.. _getting_started:

Getting Started
===============

The purpose of this guide is to demonstrate how to install starfish and process data with an
example we provide. We begin with a brief overview of starfish's data model and its expectations
for how data is formatted.

This section of the documentation describes, in order, how to install starfish, gives a brief
overview of starfish's data model and its expectations for how data is formatted, provides an
example of how to use starfish's tools to construct the index used by starfish to interact with
image data, provides instructions on loading and visualizing data, and jumps into a worked example
of constructing and applying an example data processing pipeline.
These tasks have several steps, and earlier the tutorial goes over a canonical example of an image
processing pipeline that measures the expression of some target RNA in a breast cancer tissue slice.

.. toctree::
    installation/index
    data_model/index
    formatting_data/index
    formatting_data/advanced
    loading_data/index
    example_workflow/index

# TODO ambrosejcarr: needs a comparison between smFISH and multiplex methods, otherwise later
#                    statements don't make any sense.

The following section
:ref:`Creating an Image Processing Pipeline <creating_an_image_processing_pipeline>`
goes into each of these sections in more detail.
