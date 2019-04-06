.. _loading_data:

Loading and Selecting Data
==========================

Starfish loads data by referencing the top-level :code:`experiment.json` objects in a SpaceTx-Format
dataset. The main way to load data is through the `Experiment` constructor. Assuming you've been
following the tutorial thus far and have downloaded and formatted the data from the previous
section, you can load the data as follows:

.. code-block:: python

    In [1]: import starfish

    In [2]: experiment = starfish.Experiment.from_json("iss/formatted/experiment.json")

    In [3]: experiment
    Out[3]:
    <starfish.Experiment (FOVs=2)>
    {
    fov_000: <starfish.FieldOfView>
    Primary Image: <slicedimage.TileSet (r: 4, c: 4, z: 1, x: 1390, y: 1044)>
    Auxiliary Images:
        nuclei: <slicedimage.TileSet (r: 1, c: 1, z: 1, x: 1390, y: 1044)>
        dots: <slicedimage.TileSet (r: 1, c: 1, z: 1, x: 1390, y: 1044)>
    fov_001: <starfish.FieldOfView>
    Primary Image: <slicedimage.TileSet (r: 4, c: 4, z: 1, x: 1390, y: 1044)>
    Auxiliary Images:
        nuclei: <slicedimage.TileSet (r: 1, c: 1, z: 1, x: 1390, y: 1044)>
        dots: <slicedimage.TileSet (r: 1, c: 1, z: 1, x: 1390, y: 1044)>
    }

For those who did *not* follow the tutorial, this and several other datasets are available in
the :code:`starfish.data` sub-package.

.. code-block:: python

    In [1]: import starfish
    In [2]: import starfish.data
    In [3]: experiment = starfish.data.ISS()
    In [4]: experiment
    Out[4]:
    <starfish.Experiment (FOVs=1)>
    {
    fov_001: <starfish.FieldOfView>
    Primary Image: <slicedimage.TileSet (r: 4, c: 4, x: 1390, y: 1044)>
    Auxiliary Images:
        nuclei: <slicedimage.TileSet (r: 1, c: 1, x: 1390, y: 1044)>
        dots: <slicedimage.TileSet (r: 1, c: 1, x: 1390, y: 1044)>
    }

Printing experiment shows us that we have data for two fields of view, each of which has a primary
image tensor with four rounds and channels and no z-depth. Each field of view *also* has a
corresponding image taken of the nuclei in the same spatial position, which is used to segment
cells. It also has a "dots" image, which is an image of all the spot locations, used for
registration.

An individual field of view can be extracted from the experiment as follows:

.. code-block:: python

    In [4]: fov = experiment['fov_001']

    In [5]: fov
    Out[5]:
    <starfish.FieldOfView>
    Primary Image: <slicedimage.TileSet (r: 4, c: 4, z: 1, x: 1390, y: 1044)>
    Auxiliary Images:
        nuclei: <slicedimage.TileSet (r: 1, c: 1, z: 1, x: 1390, y: 1044)>
        dots: <slicedimage.TileSet (r: 1, c: 1, z: 1, x: 1390, y: 1044)>

Simply loading the experiment and grabbing fields of view only reads the *index* into memory,
Starfish hasn't loaded any files into memory yet (or, if using an aws-localized json file, it
hasn't downloaded any images yet). Starfish enables the user to very carefully what data makes it
into the memory space of a machine.

To load a set of images, the user specifies which image they want (below we take the primary) and
can optionally specify a set of cropping parameters. To demonstrate how those work, we'll slice out
a 1000 pixel square from the :code:`(1044, 1390)` pixel :py:class:`FieldOfView`:

.. code-block:: python

    In [7]: image = fov.get_image("primary", x_slice=slice(0, 1000), y_slice=slice(0, 1000))

    In [8]: image
    Out[8]: <starfish.ImageStack (r: 4, c: 4, z: 1, y: 1000, x: 1000)>

Calling :code:`FieldOfView.get_image` localizes the data and produces an :py:class:`ImageStack`,
a 5-d tensor and *starfish*'s main in-memory image storage and processing class.

If desired, data can be further sub-selected with the :py:class:`ImageStack.sel`,

.. code-block:: python

    In [9]: from starfish.types import Axes
    In [10]: image.sel({Axes.CH: 2, Axes.ROUND: (1, 3)})
    Out[10]: <starfish.ImageStack (r: 3, c: 1, z: 1, y: 1000, x: 1000)>

Note that starfish uses constant classes for indexing so that if the SpaceTx-Format ever changed,
the same indexers could still work in starfish. Above we use the Axes constant to index into the
rounds and channels.

In addition to selection, we can max-project data, which is a commonly used filter for sparse data
to collapse :code:`z` depth into a single image tile. Here we already have non-volumetric data, so
we'll collapse all the spots across channels in each round, mimicing a "dots" image.

.. code-block:: python

    In[11]: image.max_proj(Axes.CH)
    Out[11]: <starfish.ImageStack (r: 4, c: 1, z: 1, y: 1000, x: 1000)>

Visualizing Data
----------------

For data visualization, *starfish* relies on the `napari`_ package, which is a fast image viewer
for in-memory data stored as numpy arrays. Starfish provides a wrapper over napari called
:py:func:`starfish.display`, and maintains a stable version of the package. To use the napari
viewer you must have followed the installation instructions to install the napari extra, and need
to enable the :code:`qt` environment in IPython:

.. _napari: https://github.com/napari/napari

.. code-block:: python

    In[12]: %gui qt
    In[13]: starfish.display(image)
    Out[13]: <napari.components._viewer.model.Viewer at 0x15f7b44e0>

Typing the above code should display an image viewer that looks something like this:,

#TODO ambrosejcarr annotate an image of the napari viewer.

This viewer enables the user to scroll through the rounds and channels and change the minimum and
maximum values on the colormap to visually filter the image by intensity. Later sections that deal
with spot finding
will demonstrate how :py:func:`starfish.display` can be used to visually inspect and refine the
results of spot calling.

Next, see an :ref:`Example end-to-end workflow <example_workflow>` using the starfish API.
