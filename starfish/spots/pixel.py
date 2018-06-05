import numpy as np
import pandas as pd

from starfish.munge import gather


class PixelSpotDetector:
    def __init__(self, stack):
        self.stack = stack

        self.num_objs = None
        self.spots_df = None

    def detect(self):

        sq = self.stack.squeeze()
        num_bits = self.stack.tile_metadata['barcode_index'].max() + 1

        mat = np.reshape(sq.copy(), (sq.shape[0], sq.shape[1] * sq.shape[2]))

        res = pd.DataFrame(mat.T)
        res['spot_id'] = range(len(res))
        res = gather(res, 'barcode_index', 'intensity', range(num_bits.astype(int)))
        spots_df_tidy = pd.merge(res, self.stack.tile_metadata, on='barcode_index', how='left')

        return spots_df_tidy
