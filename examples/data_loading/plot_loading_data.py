"""
.. _loading_data:

Loading Data
============

Loading Experiments
-------------------
Starfish loads data by referencing the top-level :code:`experiment.json` objects in a SpaceTx Format
dataset. The main way to load data on your machine is through the :py:class:`.Experiment`
constructor as follows:

.. code-block:: python

    In [1]: from starfish import Experiment

    In [2]: experiment = Experiment.from_json("experiment.json")

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
"""

from starfish import data

# use_test_data for loading a small subset of full dataset
experiment = data.ISS(use_test_data=True)
experiment

###################################################################################################
# In the first example, printing experiment shows us that we have data for two fields of view (FOV),
# each of which has a primary image tensor with four rounds and channels and no z-depth. Each FOV
# *also* has a corresponding image taken of the nuclei in the same spatial position, which is
# used to segment cells. It also has a "dots" image, which is an image of all the spot locations,
# used for registration.
#
# Loading Fields of View
# ----------------------
# Starfish processes FOVs separately, which enables parallel processing of large datasets. An
# individual field of view can be extracted from the experiment as follows:

fov = experiment['fov_001']
fov

###################################################################################################
# Loading Images
# --------------
# Simply loading the experiment and grabbing fields of view only reads the *index* into memory,
# Starfish hasn't loaded any files into memory yet (or, if using an aws-localized json file, it
# hasn't downloaded any images yet). Starfish enables the user to very carefully control what data
# makes it into the memory space of a machine.
#
# To load a set of images, the user specifies which image they want (below we take the primary) and
# can optionally specify a set of cropping parameters. To demonstrate how those work, we'll slice
# out a 100 pixel square from the :code:`(200, 140)` pixel :py:class:`.FieldOfView`:

image = fov.get_image("primary", x=slice(0, 100), y=slice(0, 100))
image

###################################################################################################
# Calling :py:meth:`.FieldOfView.get_image` localizes the data and produces an
# :py:class:`.ImageStack`, a 5-d tensor and *starfish*'s main in-memory image storage and
# processing class.