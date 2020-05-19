"""
.. _loading_data:

Loading and Selecting Data
==========================

Starfish loads data by referencing the top-level :code:`experiment.json` objects in a SpaceTx Format
dataset. The main way to load data on your machine is through the :py:class:`.Experiment`
constructor as follows:

.. code-block:: python

    In [1]: import starfish

    In [2]: experiment = starfish.Experiment.from_json("experiment.json")

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

To load one of the formatted example datasets in the :py:class:`.starfish.data` sub-package:

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
hasn't downloaded any images yet). Starfish enables the user to very carefully control what data
makes it into the memory space of a machine.

To load a set of images, the user specifies which image they want (below we take the primary) and
can optionally specify a set of cropping parameters. To demonstrate how those work, we'll slice out
a 1000 pixel square from the :code:`(1044, 1390)` pixel :py:class:`.FieldOfView`:

.. code-block:: python

    In [7]: image = fov.get_image("primary", x=slice(0, 1000), y=slice(0, 1000))

    In [8]: image
    Out[8]: <starfish.ImageStack (r: 4, c: 4, z: 1, y: 1000, x: 1000)>

Calling :py:meth:`.FieldOfView.get_image` localizes the data and produces an :py:class:`.ImageStack`,
a 5-d tensor and *starfish*'s main in-memory image storage and processing class.

If desired, data can be further sub-selected with the :py:meth:`.ImageStack.sel`,

.. code-block:: python

    In [9]: from starfish.types import Axes
    In [10]: image.sel({Axes.CH: 2, Axes.ROUND: (1, 3)})
    Out[10]: <starfish.ImageStack (r: 3, c: 1, z: 1, y: 1000, x: 1000)>

Note that starfish uses constant classes for indexing so that if the SpaceTx Format ever changed,
the same indexers could still work in starfish. Above we use the Axes constant to index into the
rounds and channels.

In addition to selection, we can max-project data, which is a commonly used filter for sparse data
to collapse :code:`z` depth into a single image tile. Here we already have non-volumetric data, so
we'll collapse all the spots across channels in each round, mimicking a "dots" image.

.. code-block:: python

    In[11]: from starfish.image import Filter
    In[12]: from starfish.types import FunctionSource
    In[13]: image.reduce({Axes.CH}, func="max")
    Out[13]: <starfish.ImageStack (r: 4, c: 1, z: 1, y: 1000, x: 1000)>

"""