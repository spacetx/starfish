import numpy as np
import pandas as pd


def decode(spots_df_tidy, letters=['T', 'G', 'C', 'A']):
    num_ch = spots_df_tidy.ch.max() + 1
    num_hy = spots_df_tidy.hyb.max() + 1
    num_spots = spots_df_tidy.spot_id.max() + 1

    seq_res = np.zeros((num_spots, num_hy))
    seq_stren = np.zeros((num_spots, num_hy))
    seq_qual = np.zeros((num_spots, num_hy))

    for spot_id in range(num_spots):

        sid_df = spots_df_tidy[spots_df_tidy.spot_id == spot_id]

        mat = np.zeros((num_hy, num_ch))
        inds = zip(sid_df.hyb.values, sid_df.ch.values, sid_df.val)

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
                        'qual': max_qual})

    return dec
