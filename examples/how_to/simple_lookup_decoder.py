"""
.. _howto_simplelookupdecoder:

Decoding Spots with :py:class:`.SimpleLookupDecoder`
====================================================

Linearly multiplexed assays are designed such that every RNA transcript is labeled in only one of
potentially many imaging rounds (e.g. osmFISH, sequential smFISH, and RNAscope). One way to
decode spots from images produced by these assays is to use :py:class:`.SimpleLookupDecoder`,
which simply looks up the :term:`target <Target>` in the :term:`codebook <Codebook>` whose
:term:`codeword <Codeword>` has ``value: 1`` in the round and channel the spot was found in.

.. warning::
    :py:class:`.SimpleLookupDecoder` should never be used on :py:class:`.SpotFindingResults`
    found from a ``reference_image``.

.. note::
    :py:class:`.PerRoundMaxChannel` decoding with
    ``trace_building_strategy=TraceBuildingStrategies.SEQUENTIAL`` will return effectively the
    same result but with the addition of ``xc``, ``yc``, ``zc``, ``distance``,
    and ``passes_threshold`` fields in the :py:class:`.DecodedIntensityTable`.

"""

# Load smFISH data and find spots
import starfish.data
from starfish import FieldOfView
from starfish.types import Levels
from starfish.image import Filter
experiment = starfish.data.allen_smFISH(use_test_data=True)
image = experiment["fov_001"].get_image(FieldOfView.PRIMARY_IMAGES)

bandpass = Filter.Bandpass(lshort=.5, llong=7, threshold=0.0)
glp = Filter.GaussianLowPass(
    sigma=(1, 0, 0),
    is_volume=True
)
clip1 = Filter.Clip(p_min=50, p_max=100, level_method=Levels.SCALE_BY_CHUNK)
clip2 = Filter.Clip(p_min=99, p_max=100, is_volume=True, level_method=Levels.SCALE_BY_CHUNK)
tlmpf = starfish.spots.FindSpots.TrackpyLocalMaxPeakFinder(
    spot_diameter=5,
    min_mass=0.02,
    max_size=2,
    separation=7,
    noise_size=0.65,
    preprocess=False,
    percentile=10,
    verbose=True,
    is_volume=True,
)
clip1.run(image, in_place=True)
bandpass.run(image, in_place=True)
glp.run(image, in_place=True)
clip2.run(image, in_place=True)
spots = tlmpf.run(image)

# Decode spots with SimpleLookupDecoder
from starfish.spots import DecodeSpots
decoder = DecodeSpots.SimpleLookupDecoder(codebook=experiment.codebook)
decoded_intensities = decoder.run(spots=spots)