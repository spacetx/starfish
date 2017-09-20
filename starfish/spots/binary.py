import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage.measurements as spm
from showit import image

from starfish.filters import bin_thresh
from starfish.munge import gather
from starfish.stats import label_to_regions, measure_stack


class BinarySpotDetector:
    def __init__(self, stack, thresh, blobs):
        self.stack = stack
        self.thresh = thresh

        self.blobs = blobs
        self.blobs_binary = None
        self.labels = None
        self.num_objs = None
        self.areas = None
        self.intensities = None

        self.regions = None
        self.spots_df = None

    def detect(self, measurement_type='mean'):
        self.blobs_binary = self._threshold()
        self.labels, self.num_objs = self._label()
        self.areas, self.intensities = self._measure(measurement_type)
        self.regions = self._to_regions()
        return self

    def _threshold(self):
        blobs_binary = bin_thresh(self.blobs, self.thresh)
        return blobs_binary

    def _label(self):
        labels, num_objs = spm.label(self.blobs_binary)
        return labels, num_objs

    def _measure(self, measurement_type):
        areas = spm.sum(np.ones(self.labels.shape),
                        self.labels,
                        range(1, self.num_objs))

        intensity = measure_stack(self.stack, self.labels, self.num_objs, measurement_type)
        return areas, intensity

    def _to_regions(self):
        regions = label_to_regions(self.labels)[1:]
        return regions

    def to_encoder_dataframe(self, tidy_flag):
        num_hybs = self.stack.shape[0]
        cols = range(num_hybs)
        cols = ['hyb_{}'.format(c + 1) for c in cols]
        d = dict(zip(cols, self.intensities))
        d['spot_id'] = range(self.num_objs - 1)

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
        image(self.blobs_binary, size=10, ax=plt.gca())

        plt.subplot(122)
        regions = self.regions
        image(regions.mask(background=[0.9, 0.9, 0.9],
                           dims=self.labels.shape,
                           stroke=None,
                           cmap='rainbow'),
              size=10,
              ax=plt.gca()
              )
