import numpy as np
import pandas as pd

from starfish.constants import Indices
from starfish.pipeline.features.decoded_spots import DecodedSpots
from ._base import DecoderAlgorithmBase


class IssDecoder(DecoderAlgorithmBase):
    def __init__(self, **kwargs):
        pass

    @classmethod
    def get_algorithm_name(cls):
        return "iss"

    @classmethod
    def add_arguments(cls, group_parser):
        pass

    def decode(self, intensities, codebook):  # letters=('T', 'G', 'C', 'A')):

        # TODO ambrosejcarr: fix the rest of this function
        return codebook.decode(intensities)

        # # get number of channels and hybs, as well as number of spots
        # num_ch = encoded[Indices.CH.value].max() + 1
        # num_hyb = encoded[Indices.HYB.value].max() + 1
        # num_spots = encoded['spot_id'].max() + 1
        #
        # # create some empty arrays of shape hyb * spots
        # seq_res = np.zeros((num_spots, num_hyb))
        # seq_stren = np.zeros((num_spots, num_hyb))
        # seq_qual = np.zeros((num_spots, num_hyb))
        #
        # # loop over spots
        # for spot_id in range(num_spots):
        #
        #     # get encoded spots that match spot_id
        #     sid_df = encoded[encoded.spot_id == spot_id]
        #
        #     # make a matrix of ch * hyb
        #     mat = np.zeros((num_hyb, num_ch))
        #
        #     # get an iterable of tuples of (hyb, ch, intensity)
        #     inds = zip(sid_df[Indices.HYB.value].values, sid_df[Indices.CH.value].values, sid_df['intensity'])
        #
        #     # fill the hyb * ch mat with intensities
        #     for tup in inds:
        #         mat[tup[0], tup[1]] = tup[2]
        #
        #     # get max intensity over channels for each hyb
        #     max_stren = np.max(mat, axis=1)
        #
        #     # get channel for each hyb
        #     max_ind = np.argmax(mat, axis=1)
        #
        #     # create a quality score, which is the fraction of the intensity made up by the max channel
        #     qual = max_stren / np.sum(mat, axis=1)
        #
        #     # for each spot, record the max intensity channel, its index, and quality
        #     seq_res[spot_id, :] = max_ind
        #     seq_stren[spot_id, :] = max_stren
        #     seq_qual[spot_id, :] = qual
        #
        # max_qual = np.max(seq_qual, axis=1)
        #
        # # load the codebook, and compare.
        # codes = []
        # for k in range(seq_res.shape[0]):
        #     letter_inds = seq_res[k, :]
        #     letter_inds = letter_inds.astype(np.int)
        #     res = ''.join([letters[j] for j in letter_inds])
        #     codes.append(res)
        #
        # dec = pd.DataFrame({'spot_id': range(num_spots),
        #                     'barcode': codes,
        #                     'quality': max_qual})
        #
        # # use a merge to link the codes to the codebook.
        # dec = pd.merge(dec, codebook, on='barcode', how='left')
        #
        # return DecodedSpots(dec)
