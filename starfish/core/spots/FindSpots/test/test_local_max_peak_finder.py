import sys

import numpy as np

from starfish import ImageStack
from starfish.spots import FindSpots
from starfish.types import Axes


def test_lmpf_uniform_peak():
    data_array = np.zeros(shape=(1, 1, 1, 100, 100), dtype=np.float32)
    data_array[0, 0, 0, 45:55, 45:55] = 1
    imagestack = ImageStack.from_numpy(data_array)

    # standard local max peak finder, should find spots for all the evenly illuminated pixels.
    lmpf_no_kwarg = FindSpots.LocalMaxPeakFinder(1, 1, 1, sys.maxsize)
    peaks = lmpf_no_kwarg.run(imagestack)
    results_no_kwarg = peaks[{Axes.ROUND: 0, Axes.CH: 0}]
    assert len(results_no_kwarg.spot_attrs.data) == 100

    # local max peak finder, capped at one peak per label.
    lmpf_kwarg = FindSpots.LocalMaxPeakFinder(1, 1, 1, sys.maxsize, num_peaks_per_label=1)
    peaks = lmpf_kwarg.run(imagestack)
    results_kwarg = peaks[{Axes.ROUND: 0, Axes.CH: 0}]
    assert len(results_kwarg.spot_attrs.data) == 1
