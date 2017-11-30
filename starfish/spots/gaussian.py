from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from showit import image
from skimage.feature import blob_log

from starfish.munge import gather


class GaussianSpotDetector:
    def __init__(self, stack):
        self.stack = stack

        self.blobs = None
        self.num_objs = None
        self.spots_df = None
        self.spots_df_viz = None
        self.intensities = None

    def detect(self, min_sigma, max_sigma, num_sigma, threshold, blobs, measurement_type, bit_map_flag):
        self.blobs = self.stack.aux_dict[blobs]
        fitted_blobs = self._fit(min_sigma, max_sigma, num_sigma, threshold)
        spots_df_viz = self._fitted_blobs_to_df(fitted_blobs)
        intensity = self._measure(self.blobs, spots_df_viz, measurement_type)
        spots_df_viz['intensity'] = intensity
        spots_df_viz['spot_id'] = spots_df_viz.index
        self.spots_df_viz = spots_df_viz
        self.intensities = self._measure_stack(measurement_type)
        res = self._to_encoder_dataframe(bit_map_flag)
        return res

    def _fit(self, min_sigma, max_sigma, num_sigma, threshold):
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

    def _measure(self, img, spots_df, measurement_type):
        res = []
        for row in spots_df.iterrows():
            row = row[1]
            subset = img[int(row.xmin):int(row.xmax), int(row.ymin):int(row.ymax)]

            if measurement_type == 'max':
                res.append(subset.max())
            else:
                res.append(subset.mean())

        return res

    def _measure_stack(self, measurement_type):
        intensities = [self._measure(img, self.spots_df_viz, measurement_type) for img in self.stack.squeeze()]
        return intensities

    def _to_encoder_dataframe(self, bit_map_flag):
        self.stack.squeeze(bit_map_flag=bit_map_flag)
        mapping = self.stack.squeeze_map
        inds = range(len(self.stack.squeeze_map))
        d = dict(zip(inds, self.intensities))
        d['spot_id'] = range(self.num_objs)

        res = pd.DataFrame(d)
        res = gather(res, 'ind', 'val', inds)
        res = pd.merge(res, mapping, on='ind', how='left')

        return res

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
