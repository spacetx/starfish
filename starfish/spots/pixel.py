from __future__ import division

import numpy as np
import pandas as pd

from starfish.munge import gather


class PixelSpotDetector:
    def __init__(self, stack):
        self.stack = stack

        self.num_objs = None
        self.spots_df = None

    def detect(self, bit_map_flag=False):

        sq = self.stack.squeeze(bit_map_flag=bit_map_flag)
        num_bits = self.stack.squeeze_map.bit.max() + 1

        if self.stack.is_volume:
            mat = np.reshape(sq.copy(), (sq.shape[0], sq.shape[1] * sq.shape[2] * sq.shape[3]))
        else:
            mat = np.reshape(sq.copy(), (sq.shape[0], sq.shape[1] * sq.shape[2]))

        res = pd.DataFrame(mat.T)
        res['spot_id'] = range(len(res))
        res = gather(res, 'ind', 'val', range(num_bits))
        spots_df_tidy = pd.merge(res, self.stack.squeeze_map, on='ind', how='left')

        return spots_df_tidy
