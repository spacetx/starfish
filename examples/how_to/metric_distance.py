"""
.. _howto_metricdistance:

Decoding Spots with :py:class:`.MetricDistance`
===============================================

:py:class:`.MetricDistance` is a general purpose :py:class:`.DecodeSpotsAlgorithm` that can be
used with any :term:`codebook <Codebook>` design. For
:ref:`exponentially multiplexed <tab-codebook-designs>` assays that are
*not* one hot, meaning not every round is required to have a channel with signal (e.g. MERFISH),
:py:class:`.MetricDistance` is the *only* option to decode spots. For other assays,
:py:class:`.PerRoundMaxChannel` is recommended over :py:class:`.MetricDistance` because it does
not require optimizing parameter values and has no bias introduced by the :term:`codewords
<Codeword>` in the codebook.

Unlike :py:class:`.PerRoundMaxChannel`, which constructs barcodes and then finds the matching
:term:`codeword <Codeword>`, :py:class:`.MetricDistance` transforms all :term:`codewords <Codeword>`
and :term:`spot traces <Feature (Spot, Pixel) Trace>` to a (r · c)-dimensional vectors and then maps
spot vectors to the nearest codeword vectors. Therefore, the density of the codebook can affect the
distance of spots to the nearest codewords.

For accurate decoding, it is important to :ref:`normalize<section_normalizing_intensities>` the
images prior to running :py:class:`.MetricDistance` to adjust for differences between
:term:`channel <Channel>` characteristics. During decoding, :py:class:`.MetricDistance` will also
unit normalize each vector so that spots are decoded based on the relative intensity values of
each round and channel rather than absolute intensity values.

There are a couple spot vector metrics used for filtering out poorly decoded data. The first is the
distance from :term:`target <Target>` vector calculated with the chosen distance metric. This is
stored in the :py:class:`.DecodedIntensityTable` under the ``distance`` field and can be interpreted
as a decoding quality score. The second is the vector magnitude, which is the magnitude of the
spot vector before normalizing. If either of these metrics do not pass the user-defined
thresholds then the ``passes_threshold`` value will be ``False`` in the
:py:class:`.DecodedIntensityTable`.

The example below demonstrates :py:class:`.MetricDistance` decoding on in situ sequencing data
that would normally be decoded with :py:class:`.PerRoundMaxChannel`. The parameter thresholds are
set loosely and can be tuned by analyzing spots that pass and don't pass threshold in the
:py:class`.DecodedIntensityTable`. Here the vector magnitude and distance values are plotted in a
histogram, which can provide useful information for setting thresholds.
"""

# Load in situ sequencing experiment and find spots
from starfish import data, FieldOfView
from starfish.image import ApplyTransform, LearnTransform, Filter
from starfish.types import Axes, TraceBuildingStrategies
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
    measurement_type='mean',
)
spots = bd.run(image_stack=imgs, reference_image=dots)


# Decode spots with MetricDistance set to loose parameters
from starfish.spots import DecodeSpots
decoder = DecodeSpots.MetricDistance(
    codebook=experiment.codebook,
    max_distance=1,
    min_intensity=1,
    metric='euclidean',
    norm_order=2,
    trace_building_strategy=TraceBuildingStrategies.EXACT_MATCH
)
decoded_intensities = decoder.run(spots=spots)

# Build IntensityTable with same TraceBuilder as was used in MetricDistance
from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match
intensities = build_spot_traces_exact_match(spots)
# Get vector magnitudes
norm_intensities, vector_magnitude = experiment.codebook._normalize_features(intensities, norm_order=2)
# Get distances
distances = decoded_intensities.to_decoded_dataframe().data['distance'].to_numpy()
# Plot histogram
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150
f, (ax1, ax2) = plt.subplots(ncols=2)
ax1.hist(vector_magnitude, bins=30)
ax1.set_xlabel('Barcode magnitude')
ax1.set_ylabel('Number of spots')
ax2.hist(distances, bins=30)
ax2.set_xlabel('Distance')
ax2.set_ylabel('Number of spots')
f.tight_layout
