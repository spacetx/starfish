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

        self.regions = None
        self.spots_df = None

    def detect(self):
        self.mp_thresh = self._threshold()
        self.labels, self.num_objs = self._label()
        self.areas, self.intensities = self._measure()
        self.regions = self._to_regions()
        return self

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
                        range(0, self.num_objs))

        intensity = measure_mean_stack(self.stack, self.labels, self.num_objs)
        return areas, intensity

    def _to_regions(self):
        regions = label_to_regions(self.labels)
        return regions

    def to_encoder_dataframe(self, tidy_flag):
        num_hybs = self.stack.shape[0]
        cols = range(num_hybs)
        cols = ['hyb_{}'.format(c + 1) for c in cols]
        d = dict(zip(cols, self.intensities))
        d['spot_id'] = range(self.num_objs)

        res = pd.DataFrame(d)

        if tidy_flag:
            res = gather(res, 'hybs', 'vals', cols)

        self.spots_df = res

        return self.spots_df

    def to_viz_dataframe(self):
        res = pd.DataFrame(self.regions.center)
        res = res.rename(columns=dict(zip(res.columns, ['x', 'y', 'z'])))

        if 'z' not in res.columns:
            res['z'] = None
        res['spot_id'] = res.index
        res['area'] = self.areas
        return res

    def show(self, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        plt.subplot(121)
        image(self.mp_thresh, size=10, ax=plt.gca())

        plt.subplot(122)
        regions = self.regions
        image(regions.mask(background=[0.9, 0.9, 0.9],
                           dims=self.labels.shape,
                           stroke=None,
                           cmap='rainbow'),
              size=10,
              ax=plt.gca()
              )
