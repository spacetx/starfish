from ._base import DecoderAlgorithmBase
from starfish.constants import Indices
from starfish.pipeline.features.decoded_spots import DecodedSpots


class IssDecoder(DecoderAlgorithmBase):
    def __init__(self, **kwargs):
        pass

    @classmethod
    def add_arguments(cls, group_parser):
        pass

    def decode(self, encoded, codebook, letters=('T', 'G', 'C', 'A')):
        import numpy as np
        import pandas as pd

        num_ch = encoded[Indices.CH.value].max() + 1
        num_hyb = encoded[Indices.HYB.value].max() + 1
        num_spots = encoded['spot_id'].max() + 1

        seq_res = np.zeros((num_spots, num_hyb))
        seq_stren = np.zeros((num_spots, num_hyb))
        seq_qual = np.zeros((num_spots, num_hyb))

        for spot_id in range(num_spots):

            sid_df = encoded[encoded.spot_id == spot_id]

            mat = np.zeros((num_hyb, num_ch))
            inds = zip(sid_df[Indices.HYB.value].values, sid_df[Indices.CH.value].values, sid_df['intensity'])

            for tup in inds:
                mat[tup[0], tup[1]] = tup[2]

            max_stren = np.max(mat, axis=1)
            max_ind = np.argmax(mat, axis=1)
            qual = max_stren / np.sum(mat, axis=1)

            seq_res[spot_id, :] = max_ind
            seq_stren[spot_id, :] = max_stren
            seq_qual[spot_id, :] = qual

        max_qual = np.max(seq_qual, axis=1)

        codes = []
        for k in range(seq_res.shape[0]):
            letter_inds = seq_res[k, :]
            letter_inds = letter_inds.astype(np.int)
            res = ''.join([letters[j] for j in letter_inds])
            codes.append(res)

        dec = pd.DataFrame({'spot_id': range(num_spots),
                            'barcode': codes,
                            'quality': max_qual})

        dec = pd.merge(dec, codebook, on='barcode', how='left')

        return DecodedSpots(dec)
