import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage.measurements as spm
from showit import image

from starfish.filters import bin_thresh
from starfish.munge import scale, max_proj, gather
from starfish.stats import label_to_regions, measure_mean_stack


class SimpleSpotDetector:
    def __init__(self, stack, thresh):
        self.stack = stack
        self.thresh = thresh

        self.mp_thresh = None
        self.labels = None
        self.num_objs = None
        self.areas = None
        self.intensities = None

        self.spots_df = None

    def detect(self):
        self.mp_thresh = self._threshold()
        self.labels, self.num_objs = self._label()
        self.areas, self.intensities = self._measure()

    def _threshold(self):
        mp = scale(self.stack, 'max')
        mp = max_proj(mp)
        mp_thresh = bin_thresh(mp, self.thresh)
        return mp_thresh

    def _label(self):
        labels, num_objs = spm.label(self.mp_thresh)
        return labels, num_objs

    def _measure(self):
        areas = spm.sum(np.ones(self.labels.shape),
                        self.labels,
                        np.array(range(0, self.num_objs + 1), dtype=np.int32))

        intensity = measure_mean_stack(self.stack, self.labels, self.num_objs)
        return areas, intensity

    def to_regions(self):
        regions = label_to_regions(self.labels)
        return regions

    def to_dataframe(self, tidy_flag):
        num_hybs = self.stack.shape[0]
        cols = range(num_hybs)
        cols = ['hyb_{}'.format(c + 1) for c in cols]
        d = dict(zip(cols, self.intensities))
        d['spot_id'] = range(self.num_objs)

        res = pd.DataFrame(d)

        if tidy_flag:
            res = gather(res, 'hybs', 'vals', cols)

        return res

    def show(self, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        plt.subplot(211)
        image(self.mp_thresh, bar=True, size=10, ax=plt.gca())

        plt.subplot(212)
        regions = self.to_regions()
        image(regions.mask(background=[0.9, 0.9, 0.9],
                           dims=self.labels.shape,
                           stroke=None,
                           cmap='rainbow'),
              size=10,
              ax=plt.gca()
              )
