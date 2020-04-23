"""
.. _tutorial_pixel_based_decoding:

Pixel-Based Decoding
====================

Pixel-based decoding is the approach of localizing and :term:`decoding <Decoding>` molecules
(e.g. RNA transcripts or :term:`rolonies <Rolony>`) that does not rely on algorithms to find
spots by fitting Gaussian profiles or local intensity maxima. Instead of finding spots to be
decoded, it decodes every pixel and then connects potential pixels with the same
:term:`codeword <Codeword>` into spots. The strength of this approach is it works on dense data
and noisy data where spot finding algorithms have a hard time accurately detecting spots. The
weakness is that it is prone to false positives by decoding noise that would normally be ignored
by spot finding algorithms. For this reason, pixel-based decoding is better suited for
error-robust assays such as MERFISH where the :term:`codebook <Codebook>` is designed to filter
out noise.

.. image:: /_static/design/decoding_flowchart.png
   :scale: 50 %
   :alt: Decoding Flowchart
   :align: center

Unlike :ref:`tutorial_spot_based_decoding`, which has a variety of classes and methods to
customize a workflow, pixel-based decoding is straight-forward and only requires one class in
starfish: :py:class:`.PixelSpotDecoder`. The only decoding algorithm option is the same as
:py:class:`.MetricDistance` but applied to each pixel. Normalizing images is very
important for accurate decoding, as well as vector magnitude and distance thresholds. See
:ref:`howto_metricdistance` for more details.

After decoding pixels, :py:class:`.PixelSpotDecoder` combines connected pixels into spots with
:py:func:`skimage.measure.label` and :py:func:`skimage.measure.regionprops`, which returns spot
locations and attributes. Spots that don't meet size thresholds ``min_area`` and ``max_area`` are
marked as not passing thresholds in the :py:class:`.DecodedIntensityTable`. In addition, the results
from connected component analysis that do not fit in the :py:class:`.DecodedIntensityTable` (e.g.
spot attributes and labeled images) are returned in a :py:class:`.ConnectedComponentDecodingResult`.

"""

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from starfish import data, FieldOfView, display
from starfish.image import Filter
from starfish.spots import DetectPixels
from starfish.types import Axes, Features, Levels

# Load MERFISH data
experiment = data.MERFISH(use_test_data=True)
imgs = experiment.fov().get_image(FieldOfView.PRIMARY_IMAGES)

# filter and deconvolve data
ghp = Filter.GaussianHighPass(sigma=3)
dpsf = Filter.DeconvolvePSF(num_iter=15, sigma=2, level_method=Levels.SCALE_SATURATED_BY_CHUNK)
glp = Filter.GaussianLowPass(sigma=1)
ghp.run(imgs, in_place=True)
dpsf.run(imgs, in_place=True)
glp.run(imgs, in_place=True)

# scale data with user-defined factors to normalize images. For this data set, the scale factors
# are stored in experiment.json.
scale_factors = {
    (t[Axes.ROUND], t[Axes.CH]): t['scale_factor']
    for t in experiment.extras['scale_factors']
}
filtered_imgs = deepcopy(imgs)
for selector in imgs._iter_axes():
    data = filtered_imgs.get_slice(selector)[0]
    scaled = data / scale_factors[selector[Axes.ROUND.value], selector[Axes.CH.value]]
    filtered_imgs.set_slice(selector, scaled, [Axes.ZPLANE])

# Decode with PixelSpotDecoder
psd = DetectPixels.PixelSpotDecoder(
    codebook=experiment.codebook,
    metric='euclidean',             # distance metric to use for computing distance between a pixel vector and a codeword
    norm_order=2,                   # the L_n norm is taken of each pixel vector and codeword before computing the distance. this is n
    distance_threshold=0.5176,      # minimum distance between a pixel vector and a codeword for it to be called as a gene
    magnitude_threshold=1.77e-5,    # discard any pixel vectors below this magnitude
    min_area=2,                     # do not call a 'spot' if it's area is below this threshold (measured in pixels)
    max_area=np.inf,                # do not call a 'spot' if it's area is above this threshold (measured in pixels)
)
initial_spot_intensities, prop_results = psd.run(filtered_imgs)

# filter spots that do not pass thresholds
spot_intensities = initial_spot_intensities.loc[initial_spot_intensities[Features.PASSES_THRESHOLDS]]

# Example of how to access the spot attributes
print(f"The area of the first spot is {prop_results.region_properties[0].area}")

# View labeled image after connected componenet analysis
plt.imshow(prop_results.label_image[0])
plt.title("PixelSpotDecoder Labeled Image")

# View decoded spots overlaid on max intensity projected image
single_plane_max = filtered_imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")
# Uncomment code below to view spots
#%gui qt
#viewer = display(stack=single_plane_max, spots=spot_intensities)