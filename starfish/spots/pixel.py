import numpy as np
import pandas as pd

from starfish.munge import melt


class PixelSpotDetector:
    def __init__(self, stack):
        self.stack = stack

        self.num_objs = None
        self.spots_df = None

    def detect(self):

        sq = self.stack.squeeze()
        num_bits = int(self.stack.tile_metadata['barcode_index'].max() + 1)

        # linearize the pixels, mat.shape = (n_hybs * n_channels * n_z_slice, x * y)
        mat = np.reshape(sq.copy(), (sq.shape[0], sq.shape[1] * sq.shape[2]))

        res = pd.DataFrame(mat.T)
        res['spot_id'] = range(len(res))
        res = melt(
            df=res,
            new_index_name='barcode_index',
            new_value_name='intensity',
            melt_columns=range(num_bits)
        )
        spots_df_tidy = pd.merge(res, self.stack.tile_metadata, on='barcode_index', how='left')

        return spots_df_tidy
