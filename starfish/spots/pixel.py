from __future__ import division

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu


class PixelSpotDetector:
    def __init__(self, stack):
        self.stack = stack

        self.num_objs = None
        self.spots_df = None
        self.spots_df_viz = None
        self.threshold = None

    def detect(self):
        self.spots_df_viz = self.to_viz_dataframe()
        self.spots_df = self.to_encoder_dataframe()

    def to_encoder_dataframe(self):
        """
        Writes table of spot_id | hyb | ch | val
        :param min_intensity: minimum intensity of pixel value to retain 
        :return: spots_df 
        """

        def encode(row, squeezed_stack):
            r = row[1]
            sq = squeezed_stack[r.ind]
            sqt = sq[self.threshold[:, 0], self.threshold[:, 1]]
            val = sqt.ravel()
            d = pd.DataFrame()
            d['spot_id'] = range(self.num_objs)
            d['val'] = val
            d['hyb'] = r.hyb
            d['ch'] = r.ch
            return d

        squeezed_stack = self.stack.squeeze()
        squeeze_map = self.stack.squeeze_map
        res = [encode(row, squeezed_stack) for row in squeeze_map.iterrows()]
        self.spots_df = pd.concat(res)

        return self.spots_df

    def to_viz_dataframe(self):
        """
        Determines pixel locations worthy of decoding. Uses otsu thresholding to separate
        signal from background. This decreases search space necessary for pixel based readout
        :return: spots_df_viz: | spot_id | x | y | z | 
        """

        # determine threshold
        mp = self.stack.max_proj('hyb')
        mp = np.max(mp, axis=0)
        t = threshold_otsu(mp)
        ind = np.argwhere(mp > t)
        self.num_objs = np.sum(mp > t)
        self.threshold = ind

        # construct spots_viz
        self.num_objs = ind.shape[0]
        df = pd.DataFrame()
        df['spot_id'] = range(self.num_objs)
        df['x'] = ind[:, 0]
        df['y'] = ind[:, 1]
        self.spots_df_viz = df

        return self.spots_df_viz
