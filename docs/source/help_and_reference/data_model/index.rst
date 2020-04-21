.. _data_model:

Data Model
==========

The starfish package is designed to allow users to create pipelines to count spots and
assign those spots to cells. In order to accomplish this, starfish defines a series of
abstractions, and those abstractions in turn define how we require the data to be formatted,
and how the data is processed.

Imaging experiments that are compatible with starfish typically cannot capture an entire tissue
slide within the field of view of a microscope. As a result, these experiments are often captured as
collages of sometimes-overlapping images that are stitched back into panoramas with software.

Because the complete images are often tens of thousands of pixels in :code:`x` and :code:`y`, it
can be challenging to process panoramas. In contrast, fields of view are often small enough
that they can be processed on laptops. As a result, panoramas are stored in :code:`Experiment`
objects, which are composed of all the fields of view captured by the microscope.

Note that at the current time, this means there is some up-front cost to users, who will need to
determine, using our documentation, how to reformat their microscopy data. We are working with
`Bio-Formats <bio_formats>`_ to automate this conversion and reduce this up-front cost.

Field of View
-------------

A field of view in a starfish-compatible experiment contains images of several types. Each field
of view **must** contain images containing spots. These are the "primary" images. Additional images
with the same :code:`(y, x)` size of nuclei (for segmenting cells) or fiduciary markers (for
registering images within the field of view) can also be associated with a field of view, if the
user captures them. It is important to note that all images regardless of the fluorescence channel,
time point/round, or z-plane that are taken of the specific :code:`(y, x)` coordinate area should
be included in a field of view:

.. image:: /_static/design/imagestack.png

This data structure is general across the types of data that we've observed in image-based
transcriptomics and proteomics studies, which have variables numbers of fluorescence channels
and imaging rounds. Starfish simply builds up a set of 5D tensor for each field of view, one for
each image type (primary, dots, nuclei, ...). The dimensions are :code:`round (time), channel,
z, y, x`.

.. image:: /_static/design/field_of_view.png

Processing Model
----------------

Starfish breaks up data by field of view to enable efficient processing. Unlike stitched images,
fields of view are often small enough to work with on laptops, and are efficient to pass around to
cloud compute instances or HPC systems. They're also more easily broken up into pieces that fit on
modern GPUs, which while not yet integrated in starfish, show promise for speeding early image
processing tasks.

This processing model is particularly amenable to processing on the cloud, where there is no joint
file system that all the compute instances can access. In these ecosystems, input data for pipelines
must be localized and downloaded to the machine, and uploaded back to the data store when processing
is completed. By working with small 2-dimensional planes, starfish is able to exercise granular
control over data upload, producing an efficient processing system for large imaging experiments
that works both on local HPC clusters and on the cloud.

.. image:: /_static/design/processing_model.png

Dual Coordinate Systems
-----------------------

Because starfish requires that microscopy experiments are broken up into fields of view, it is
important to keep track of the physical bounding box of each field of view. This information will be
needed to convert data into SpaceTx format, and outputs that starfish produces will track objects
in this coordinate space.

Internally, starfish treats images as tensors, and will primarily operate on pixel coordinates,
bringing the physical coordinates along for the ride. This duality will pervade the remainder of
the documentation.


Next Steps
----------

At this point, the tutorial forks. You can either dive into formatting your data in SpaceTx-Format,
or play skip forward and play with starfish using our pre-constructed example datasets.
We suggest the latter, as it will give you a sense of starfish's capabilities before you put work
into reformatting your data.
