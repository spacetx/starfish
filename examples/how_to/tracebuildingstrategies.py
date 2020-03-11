"""
.. _howto_tracebuildingstrategies:

When to Use Each ``TraceBuildingStrategies``
============================================

In order to multiplex image-based transcriptomics assays beyond the number of spectrally distinct
fluorophores, assays use multiple rounds of imaging. Before every round, RNA transcripts or
amplicons are relabeled so that if you trace a spot across rounds, the spot will have a
pattern of signals. This pattern should match a :term:`codeword<Codeword>` in the
:term:`codebook<Codebook>`.

There are three different ``TraceBuilders`` that can be used to trace spots in
:py:class:`.SpotFindingResults` into an :py:class:`.IntensityTable` or
:py:class:`.DecodedIntensityTable`. It is important to choose the correct ``TraceBuilder`` that
matches the codebook design and data.

.. image:: /_static/design/tracebuilder_decisiontree.png
   :scale: 50 %
   :alt: Which TraceBuilder To Use
   :align: center

The chosen ``TraceBuilder`` must also be compatible with how the :py:class:`.SpotFindingResults`
was generated. :py:class:`.FindSpotsAlgorithm`\s can be run with or without a ``reference_image``.
If run with a ``reference_image`` then every :py:class:`.PerImageSliceSpotResults` in
:py:class:`.SpotFindingResults` will have the same spots for every (round, channel) image volume.
This is necessary for :py:func:`.build_spot_traces_exact_match` but not recommended for
:py:func:`.build_traces_sequential` and :py:func:`.build_traces_nearest_neighbors`.

.. list-table:: ``TraceBuildingStrategy``
   :widths: auto
   :header-rows: 1

   * - Method
     - Description
     - Reference Image
   * - ``SEQUENTIAL``
     - Build traces for every detected spot by setting intensity values to zero for all rounds
       and channels the spot was not found in (i.e. every trace will have only 1 non-zero value)
     - Incompatible
   * - ``EXACT_MATCH``
     - Build traces by combining intensity values of spots from every rounds and channel in the
       exact same location as spots in ``reference_image``
     - Required
   * - ``NEAREST_NEIGHBOR``
     - Build traces by combining intensity values of spots from rounds and channels nearest to the
       spots in the ``anchor_round``
     - Not recommended; will have same result as EXACT_MATCH

"""

# Load and process ISS images to find spots with and without reference image
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

# run blob detector on dots and on image stack
bd = FindSpots.BlobDetector(
    min_sigma=1,
    max_sigma=10,
    num_sigma=30,
    threshold=0.01,
    measurement_type='mean',
)
dots_max = dots.reduce((Axes.ROUND, Axes.ZPLANE), func="max")
spots_from_ref = bd.run(image_stack=imgs, reference_image=dots_max)
spots_from_stack = bd.run(image_stack=imgs)


####################################################################################################
# Typical pipelines will set the ``trace_building_strategy`` as an argument in the
# :py:class:`.DecodeSpotsAlgorithm` but here the underlying code is exposed to reveal what the
# different :py:class:`.IntensityTable`\s look like depending on which ``TraceBuilder`` is used.

from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match, \
    build_traces_sequential, build_traces_nearest_neighbors

print('Build trace with EXACT_MATCH')
print(build_spot_traces_exact_match(spots_from_ref))

####################################################################################################
# When building spot traces with EXACT_MATCH, every feature has a value in each round and channel
# because a ``reference_image`` was used in spot finding.

print('\nBuild trace with SEQUENTIAL')
print(build_traces_sequential(spots_from_stack))

####################################################################################################
# When building spot traces with SEQUENTIAL, every feature has only one non-zero round and channel
# because :py:func:`.build_traces_sequential` automatically assigns a zero value to all other
# rounds and channels.

print('\nBuild trace with NEAREST_NEIGHBORS')
print(build_traces_nearest_neighbors(spots_from_stack, search_radius=5))

####################################################################################################
# When building spot traces with NEAREST_NEIGHBORS on spots found in :py:class:`.ImageStack`
# without a ``reference image``, the ``nan`` values are due to no spot being found within the
# ``search_radius`` of the ``anchor_round``.

print('\nBuild trace with NEAREST_NEIGHBORS')
print(build_traces_nearest_neighbors(spots_from_ref, search_radius=5))

####################################################################################################
# The same :py:func:`.build_traces_nearest_neighbors` applied to spots found in
# :py:class:`.ImageStack` *with* a ``reference image`` guarantees a spot to be found in every
# round of :py:class:`.SpotFindingResults`.