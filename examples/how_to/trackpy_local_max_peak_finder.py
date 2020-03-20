"""
.. _howto_trackpylocalmaxpeakfinder:

Finding Spots with :py:class:`.TrackpyLocalMaxPeakFinder`
=========================================================

RNA FISH spots are usually bright point spread functions in a greyscale image.
:term:`Rolonies<Rolony>`, which are rolling-circle amplicons produced in certain assays (e.g. in
situ sequencing), are approximately 1-um diameter Gaussian spots. Generally, the recommended
:py:class:`.FindSpotsAlgorithm` to use in a starfish pipeline is :py:class:`.BlobDetector`
because it accounts for the intensity profile of a spot rather than just thresholding pixel
values. But for some images :py:class:`.BlobDetector` may not be satisfactory so starfish also
provides alternatives.

:py:class:`.TrackpyLocalMaxPeakFinder` finds Gaussian spots by an implementation of the
Crocker-Grier centroid-finding algorithm. Local maxima are treated as centroids and then the
locations are refined to obtain sub-pixel accuracy. Unlike :py:class:`.LocalMaxPeakFinder`,
:py:class:`.TrackpyLocalMaxPeakFinder` measures spot attributes like size and eccentricity but it
also requires the user to manually set a minimum intensity threshold.

Before running :py:class:`.TrackpyLocalMaxPeakFinder`, the image must be preprocessed to
smooth noise and remove background. This can be done by setting ``preprocess=True`` or by running
:py:class:`.Bandpass` filter separately. If done in :py:class:`.TrackpyLocalMaxPeakFinder`,
``noise_size`` defines the width of the Gaussian blurring kernel for removing high frequency
noise and ``smoothing_size`` defines the width of boxcar smoothing kernel. There is no formula
for picking ``noise_size``, but larger values will lead to more blurring and it should be less
than ``smoothing_size``. And ``smoothing_size`` should generally be the diameter of a spot
rounded up to the nearest odd integer. The background should be set to zero with a ``threshold``
set in  :py:class:`.Bandpass` or by
ref:`clipping background to zero  <howto_clip_percentile_to_zero>`.

For finding spots, ``spot_diameter`` is used to identify all possible spots and should be set to
the diameter of the spots on zero background rounded up to the nearest odd integer. Then
thresholds such as ``min_mass`` (integrated brightness), ``max_size`` (radius of gyration),
``percentile`` (relative peak brightness), and ``separation`` (distance between spots) are used
to filter out spots. Plotting these attributes can help ballpark cutoff values but :ref:`visually
assessing <howto_spotfindingresults>` results is needed to validate the parameter settings before
running in batch.

.. warning::
    :py:class:`.TrackpyLocalMaxPeakFinder` does not support finding spots on
    independent 2D slices of a volume (i.e., ``is_volume = False``).

.. warning::
    :py:class:`.TrackpyLocalMaxPeakFinder` is not compatible with cropped data sets.

"""

from starfish import data
from starfish import FieldOfView
from starfish.image import Filter
from starfish.spots import FindSpots

experiment = data.allen_smFISH(use_test_data=True)
img = experiment['fov_001'].get_image(FieldOfView.PRIMARY_IMAGES)

# filter to remove noise, remove background, blur, and clip
bandpass = Filter.Bandpass(lshort=.5, llong=7, threshold=0.0)
glp = Filter.GaussianLowPass(
    sigma=(1, 0, 0),
    is_volume=True
)
clip1 = Filter.Clip(p_min=50, p_max=100)
clip2 = Filter.Clip(p_min=99, p_max=100, is_volume=True)
clip1.run(img, in_place=True)
bandpass.run(img, in_place=True)
glp.run(img, in_place=True)
clip2.run(img, in_place=True)


tlmpf = FindSpots.TrackpyLocalMaxPeakFinder(
    spot_diameter=5,  # must be odd integer
    min_mass=0.02,
    max_size=2,  # this is max radius
    separation=7,
    preprocess=False,
    percentile=10,  # this has no effect when min_mass, spot_diameter, and max_size are set properly
    verbose=True,
)
spots = tlmpf.run(img)
