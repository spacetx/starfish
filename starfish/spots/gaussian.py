from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from showit import image
from skimage.feature import blob_log

from starfish.munge import stack_to_list, gather


class GaussianSpotDetector:
    def __init__(self, stack, blobs, measurement_type='max'):
        self.stack = stack
        self.blobs = blobs
        self.measurement_type = measurement_type

        self.num_objs = None
        self.spots_df = None
        self.spots_df_viz = None
        self.intensities = None

    def detect(self, min_sigma, max_sigma, num_sigma, threshold):
        fitted_blobs = self.fit(min_sigma, max_sigma, num_sigma, threshold)
        spots_df_viz = self._fitted_blobs_to_df(fitted_blobs)
        intensity = self._measure(self.blobs, spots_df_viz)
        spots_df_viz['intensity'] = intensity
        spots_df_viz['spot_id'] = spots_df_viz.index
        self.spots_df_viz = spots_df_viz
        self.intensities = self._measure_stack()

    def fit(self, min_sigma, max_sigma, num_sigma, threshold):
        fitted_blobs = blob_log(self.blobs, min_sigma, max_sigma, num_sigma, threshold)
        fitted_blobs[:, 2] = fitted_blobs[:, 2] * np.sqrt(2)
        self.num_objs = fitted_blobs.shape[0]
        return fitted_blobs

    @staticmethod
    def _fitted_blobs_to_df(fitted_blobs):
        res = pd.DataFrame(fitted_blobs)
        res['d'] = res[2] * 2
        res.rename(columns={0: 'x', 1: 'y', 2: 'r'}, inplace=True)
        res['xmin'] = np.floor(res.x - res.r)
        res['xmax'] = np.ceil(res.x + res.r)
        res['ymin'] = np.floor(res.y - res.r)
        res['ymax'] = np.ceil(res.y + res.r)
        return res

    def _measure(self, img, spots_df):
        res = []
        for row in spots_df.iterrows():
            row = row[1]
            subset = img[int(row.xmin):int(row.xmax), int(row.ymin):int(row.ymax)]
            if self.measurement_type == 'max':
                try:
                    res.append(subset.max())
                except:
                    res.append(np.NAN)
            else:
                try:
                    res.append(subset.mean())
                except:
                    res.append(np.NAN)
        return res

    def _measure_stack(self):
        intensities = [self._measure(img, self.spots_df_viz) for img in stack_to_list(self.stack)]
        return intensities

    def to_encoder_dataframe(self, tidy_flag, mapping=None):
        inds = range(self.stack.shape[0])
        d = dict(zip(inds, self.intensities))
        d['spot_id'] = range(self.num_objs)

        res = pd.DataFrame(d)

        if tidy_flag:
            res = gather(res, 'ind', 'val', inds)
            if mapping is not None:
                res = pd.merge(res, mapping, on='ind', how='left')
                del res['ind']

        self.spots_df = res

        return self.spots_df

    def to_viz_dataframe(self):
        return self.spots_df_viz

    def show(self, figsize=(20, 20)):
        plt.figure(figsize=figsize)
        ax = plt.gca()

        image(self.blobs, ax=ax)
        blobs_log = self.spots_df_viz.loc[:, ['x', 'y', 'r']].values
        for blob in blobs_log:
            x, y, r = blob
            c = plt.Circle((y, x), r, color='r', linewidth=2, fill=False)
            ax.add_patch(c)

        plt.title('Num blobs: {}'.format(len(blobs_log)))
        plt.show()
