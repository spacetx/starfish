"""
.. _plot_display:

Visualizing Data
================

For data visualization, *starfish* relies on the `napari`_ package, which is a fast image viewer
for in-memory data stored as numpy arrays. Starfish provides a wrapper over napari called
:py:func:`.display`, and maintains a stable version of the package. To use the napari
:code:`Viewer` you must have followed :ref:`installation` to install the napari extra
and enabled the :code:`qt` environment in IPython.

.. _napari: https://github.com/napari/napari

The :py:func:`.display` function can be used to visually inspect your raw images, filtered
images, results of spot finding, and cell segmentation masks. All of these can be overlaid on the
same coordinates as layers that can be toggled on/off, making it extremely easy to compare the
inputs and outputs of starfish functions. The table below shows which starfish data structures
can be viewed with :py:func:`.display` and how.

.. table::
   :class: "table-bordered"

   +-----------------------------------+-------------------+-------------------+
   | Data Structure                    | display parameter | napari layer type |
   +===================================+===================+===================+
   | :py:class:`.ImageStack`           | stack             | image             |
   +-----------------------------------+-------------------+-------------------+
   | :py:class:`.IntensityTable`       | spots             | points            |
   +-----------------------------------+-------------------+-------------------+
   | :py:class:`.BinaryMaskCollection` | masks             | labels            |
   +-----------------------------------+-------------------+-------------------+

.. note::

    Currently starfish does not support displaying :py:class:`.SpotFindingResults` directly
    (see: `#1721`_). One option is to use a
    :ref:`TraceBuildingStrategy<howto_tracebuildingstrategies>` or a
    :ref:`DecodeSpotsAlgorithm<spot_decoding_table>` to create an
    :py:class:`.IntensityTable` or `:py:class:`.DecodedIntensityTable`, respectively. But if you
    need to directly view the results of your spot finding, see the last section of this tutorial.

.. _#1721: https://github.com/spacetx/starfish/issues/1721

Basic examples of visualizing data with :py:func:`.display()` can be found in :ref:`display` and
application-specific examples are scattered throughout the tutorials. This tutorial will
demonstrate some more advanced functions of the image :code:`Viewer` using napari commands.
Due to the GUI, the code cannot be rendered here, but you can use the same code in your IPython
console.
"""

###################################################################################################
# Start with the :ref:`quick start` tutorial
# ------------------------------------------
# For this demonstration, we will use the :ref:`quick start` data :code:`registered_imgs`,
# :code:`decoded`, and :code:`masks`, which are an :py:class:`.ImageStack`,
# :py:class:`.DecodedIntensityTable`, and :py:class:`.BinaryMaskCollection`, respectively.
#
# If you run :py:func:`.display` like so:
#
# >>> %gui qt
# >>> display(stack=registered_imgs, spots=decoded, masks=masks)
#
# An image :code:`Viewer` should pop up that looks something like this:
#
# .. image:: /_static/design/napari-viewer.png

###################################################################################################
# Using the GUI
# -------------
# We recommend, checking out the napari `viewer`_ tutorial to see everything you can do with
# the GUI but some basic functions you can try include using your scroll wheel to zoom,
# sliding the sliders to view different rounds and channels, and toggling the layer visibilities.
#
# .. _viewer: https://napari.org/tutorials/fundamentals/viewer

###################################################################################################
# Adding additional layers
# ------------------------
# Now you may want to view more than one of each data structure in the same :code:`Viewer`. For
# example, to compare an :py:class:`.ImageStack` before and after filtering. :py:func:`.display`
# only accepts one argument for each parameter, so to have two layers of the same type you have to
# call :py:func:`.display` a second time and pass it the name of the first image :code:`Viewer`.
#
# >>> viewer = display(stack=registered_imgs, spots=decoded, masks=masks)
# >>> viewer = display(stack=imgs_wth, viewer=viewer)
#
# This adds another layer named :code:`stack [1]` to differentiate it from the original
# :code:`stack`.
#
# .. image:: /_static/images/quickstart-napari-screenshot-3.png

###################################################################################################
# Naming your layers with napari commands
# ---------------------------------------
# By default, :py:func:`.display` names every :py:class:`.ImageStack` layer :code:`stack`,
# every :py:class:`.IntensityTable` layer :code:`spots`, and every :py:class:`.BinaryMaskCollection`
# layer :code:`masks`. If you want to give the layers custom names, you could use the GUI and double
# click the names to edit. Or you can use napari methods, which gives you many more options to
# tweak your layer properties (see `napari`_).
#
# However, napari doesn't natively support starfish data structures, so to use napari methods
# follow the example below. Adding :py:class:`.IntensityTable`\s is pretty complex,
# so we recommend sticking with :py:func:`.display` for that or using the method in the next
# section.
#
# >>> import napari
# >>> viewer.add_image(dots_wth.xarray, name='dots')
# >>> viewer.add_labels(masks.to_label_image().label_image, name='cells')
#
# .. image:: /_static/images/quickstart-napari-screenshot-4.png

###################################################################################################
# Directly visualizing :py:class:`.SpotFindingResults`
# ----------------------------------------------------
# You can convert the :py:class:`.SpotAttributes` from each round and channel of
# :py:class:`.SpotFindingResults` to a numpy array and then use napari commands to display them
# in your :code:`Viewer`.
#
# .. code-block:: python
#
#   # save SpotAttributes for each (round,channel) as numpy array in list
#   spots_numpy = list()
#   for rnd in spots.round_labels:
#       for ch in spots.ch_labels:
#           spots_numpy.append(spots[{Axes.CH:ch, Axes.ROUND:rnd}].spot_attrs.data[['z', 'y',
#           'x']].to_numpy())
#
#   # display found spots for each (round,channel) as a layer in napari
#   viewer = display(stack=imgs)
#   layer_index = 0
#   for rnd in spots.round_labels:
#       for ch in spots.ch_labels:
#           viewer.add_points(data=spots_numpy[layer_index], symbol='ring', face_color='red',
#                             size=5, name=f'r: {rnd}, ch: {ch}', visible=False)
#           layer_index = layer_index + 1
#
# .. image:: /_static/images/quickstart-napari-screenshot-5.png
