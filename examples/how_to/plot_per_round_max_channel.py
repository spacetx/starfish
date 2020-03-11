"""
.. _howto_perroundmaxchannel:

Decoding Spots with :py:class:`.PerRoundMaxChannel`
===================================================

:py:class:`.PerRoundMaxChannel` is a :py:class:`.DecodeSpotsAlgorithm` that picks the channel with
maximum signal intensity for each round to construct a barcode and then matches the barcode to
:term:`codewords <Codeword>` in the :term:`codebook <Codebook>`. It is important to
:ref:`normalize<section_normalizing_intensities>` the images prior to
:py:class:`.PerRoundMaxChannel` if the channels have significant differences in range of
intensity values. The returned :py:class:`.DecodedIntensityTable` has a ``distance`` field that
is a decoding quality score. :term:`Spots traces <Feature (Spot, Pixel) Trace>` with higher signal
in non-max channels have a greater ``distance`` value reflecting lower confidence in the decoded
:term:`target <Target>`.

:py:class:`.PerRoundMaxChannel` can be used for linearly multiplexed and one hot multiplexed
:term:`codebooks <Codebook>`. Linearly multiplexed assays (e.g. osmFISH, sequential
smFISH, and RNAscope) can be decoded with :py:class:`.PerRoundMaxChannel` by setting
``trace_building_strategy=TraceBuildingStrategies.SEQUENTIAL``. One hot multiplexed assays (e.g.
in situ sequencing, seqFISH, and STARmap) are termed 'one hot' because every round has exactly one
hot channel. They can be decoded with :py:class:`.PerRoundMaxChannel` by setting
``trace_building_strategy=TraceBuildingStrategies.EXACT_MATCH`` or
``trace_building_strategy=TraceBuildingStrategies.NEAREST_NEIGHBORS``. The example below
demonstrates the recommended method for decoding one hot multiplexed
data using :py:class:`.PerRoundMaxChannel`.
"""

# Load in situ sequencing experiment and find spots
from starfish.image import ApplyTransform, LearnTransform, Filter
from starfish.types import Axes, TraceBuildingStrategies
from starfish import data, FieldOfView
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
    max_sigma=10,
    num_sigma=30,
    threshold=0.01,
    measurement_type='mean',
)
dots_max = dots.reduce((Axes.ROUND, Axes.ZPLANE), func="max")
spots = bd.run(image_stack=imgs, reference_image=dots_max)

# Decode spots with PerRoundMaxChannel
from starfish.spots import DecodeSpots
decoder = DecodeSpots.PerRoundMaxChannel(
    codebook=experiment.codebook,
    trace_building_strategy=TraceBuildingStrategies.EXACT_MATCH
)
decoded_intensities = decoder.run(spots=spots)