"""
.. _howto_blobdetector:

Finding Spots with :py:class:`.BlobDetector`
============================================

RNA FISH spots are usually bright point spread functions in a greyscale image.
:term:`Rolonies<Rolony>`, which are rolling-circle amplicons produced in certain assays (e.g. in
situ sequencing), are approximately 1-um diameter Gaussian spots. Despite their minor differences in
size and intensity profiles, these "blobs" can be detected using a common computer vision
technique that convolves kernels with an image to identify where the blobs are. The kernels,
or filters as they are sometimes called, find spots that are the same size as it.

Starfish implements this blob detection technique with :py:class:`.BlobDetector`. It supports three
`approaches <https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html>`_
that can be chosen with the ``detector_method`` argument when instantiating the
detector. The default LoG approach produces the most accurate results and should be used unless
computation time becomes a concern.

In order to detect spots of various sizes in the same set of images, :py:class:`.BlobDetector`
convolves kernels of multiple sizes and picks the best fit for each spot. The kernel sizes are
defined by sigma, which is the standard deviation of the Gaussian used in each approach. The user
must define the range of sigmas to be used with the ``min_sigma``, ``max_sigma`` and ``num_sigma``
arguments. Picking the right sigma requires looking at the images and approximating the size of
spots. Using a wider range of sizes and increasing ``num_sigma`` can find more spots but will
require more computation time and possibly capture noise that is not the correct size as true RNA
spots. That is why it is recommended to only choose sigmas that make sense for the experiment and
microscope settings. The table below is a helpful guide for setting the sigma parameters based on
radii of spots.

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Approach
     - Sigma =
   * - LoG 2D
     - radius / sqrt(2)
   * - LoG 3D
     - radius / sqrt(3)
   * - DoG 2D
     - radius / sqrt(2)
   * - DoG 3D
     - radius / sqrt(3)
   * - DoH
     - radius

Another parameter of :py:class:`.BlobDetector` is ``threshold``, which filters out spots with
low intensities that are likely background noise. One way to set ``threshold`` is to choose a
conservatively low value to start with on a representative image and :ref:`visually assess
<howto_spotfindingresults>` results. If the image has a high SNR the ``threshold`` is trivial but
if there is high background, then choosing the right ``threshold`` value can become subjective.
Another way to estimate ``threshold`` is :ref:`howto_localmaxpeakfinder` and examining the
intensities of the spots.

.. warning::
    :py:class:`.BlobDetector` is not compatible with cropped data sets.

"""

# Load in situ sequencing experiment
from starfish.image import ApplyTransform, LearnTransform, Filter
from starfish.types import Axes
from starfish import data, FieldOfView
from starfish.spots import FindSpots
from starfish.util.plot import imshow_plane
experiment = data.ISS()
fov = experiment.fov()
imgs = fov.get_image(FieldOfView.PRIMARY_IMAGES) # primary images
dots = fov.get_image("dots") # reference round where every spot labeled with fluorophore

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

# view dots to estimate radius of spots: radius range from 1.5 to 4 pixels
imshow_plane(dots, {Axes.X: (500, 550), Axes.Y: (600, 650)})

# run blob detector with dots as reference image
# following guideline of sigma = radius/sqrt(2) for 2D images
# threshold is set conservatively low
bd = FindSpots.BlobDetector(
    min_sigma=1,
    max_sigma=3,
    num_sigma=10,
    threshold=0.01,
    is_volume=False,
    measurement_type='mean',
)
spots = bd.run(image_stack=imgs, reference_image=dots)