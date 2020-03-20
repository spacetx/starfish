"""
.. _howto_spotfindingresults:

Assessing :py:class:`.SpotFindingResults`
=========================================

Purpose of this tutorial:

* Deciding between spot-based or pixel-based decoding
* Choosing a :py:class:`.FindSpotsAlgorithm`
* Tuning a :py:class:`.FindSpotsAlgorithm`

Although it is not necessary to visualize spots found by the :py:class:`.FindSpotsAlgorithm` before
decoding every field of view in your data, it can be a useful step when building an image
processing pipeline. Visually assessing the detected spots will ensure the spot-based decoding
approach and :py:class:`.FindSpotsAlgorithm` you chose is optimized for your data. To learn more
about how spots can be found and decoded in starfish see :ref:`section_finding_and_decoding`.

There are two methods for viewing spots. The first is to access the :py:class:`.SpotAttributes`
of a selected :term:`ImageSlice` and add it as points to the napari viewer. The second is to use a
``TraceBuilder`` to convert the :py:class:`.SpotFindingResults` to an
:py:class:`.IntensityTable`, which can then be passed to :py:func:`.display`.

.. note::
    :py:class:`.DecodedIntensityTable` can also be passed to :py:func:`.display`.

"""

# Load and process ISS images to find spots with BlobDetector
from starfish.image import ApplyTransform, LearnTransform, Filter
from starfish.types import Axes
from starfish import data, display, FieldOfView
from starfish.spots import FindSpots

experiment = data.ISS()
fov = experiment.fov()
imgs = fov.get_image(FieldOfView.PRIMARY_IMAGES) # primary images
dots = fov.get_image("dots") # reference round for image registration

# filter raw data
masking_radius = 15
filt = Filter.WhiteTophat(masking_radius, is_volume=False)
filt.run(imgs, in_place=True)
filt.run(dots, in_place=True)

# register primary images to reference round
learn_translation = LearnTransform.Translation(reference_stack=dots, axes=Axes.ROUND, upsampling=1000)
transforms_list = learn_translation.run(imgs.reduce({Axes.CH, Axes.ZPLANE}, func="max"))
warp = ApplyTransform.Warp()
warp.run(imgs, transforms_list=transforms_list, in_place=True)

# run blob detector on dots (reference image with every spot)
bd = FindSpots.BlobDetector(
    min_sigma=1,
    max_sigma=3,
    num_sigma=10,
    threshold=0.01,
    is_volume=False,
    measurement_type='mean',
)
spots = bd.run(image_stack=imgs, reference_image=dots)

####################################################################################################
# The first way to visualize detected spots is to access the :py:class:`.SpotAttributes`. Since
# spots were found using a reference image, the :py:class:`.SpotAttributes` for every
# :term:`ImageSlice` in :py:class:`.SpotFindingResults` is the same and it doesn't matter
# which ImageSlice is selected to display. If no reference image were passed to
# :py:meth:`.BlobDetector.run`, then each ImageSlice would contain different
# :py:class:`.SpotAttributes` and it would be best to display each as a different points layer to
# be compared with the :py:class:`.ImageStack`.

# uncomment code to view
# %gui qt
# viewer = display(stack=dots)
# viewer.add_points(data=spots[{Axes.CH:1, Axes.ROUND:0}].spot_attrs.data[['z', 'y',
# 'x']].to_numpy(), size=5)

####################################################################################################
# The other way to visualize detected spots is to convert the :py:class:`.SpotFindingResults` to
# an :py:class:`.IntensityTable`. This can be done by decoding to a
# :py:class:`.DecodedIntensityTable`, which is a subclass of :py:class:`.IntensityTable`.
# However, a :py:class:`.Codebook` independent method is to use a ``TraceBuilder`` to return an
# :py:class:`.IntensityTable`. See :ref:`howto_tracebuildingstrategies` to pick the suitable
# ``TraceBuilder``.

from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match
intensity_table = build_spot_traces_exact_match(spots)

# uncomment code to view
# %gui qt
# viewer = display(stack=dots, spots=intensity_table)
