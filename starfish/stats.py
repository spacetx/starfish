import numpy as np
import pandas as pd
import scipy.ndimage.measurements as spm
from regional import many as Many
from regional import one as One
from scipy.sparse import coo_matrix

from starfish.munge import stack_to_list


def stack_describe(stack):
    num_hybs = stack.shape[0]
    stats = [im_describe(stack[k, :]) for k in range(num_hybs)]
    return stats


def im_describe(im):
    shape = im.shape
    flat_dims = reduce(lambda x, y: x * y, shape)
    flat_im = np.reshape(im, flat_dims)
    stats = pd.Series(flat_im).describe()
    return stats.to_dict()


def regions_stat(label_mat):
    label_mat_coo = coo_matrix(label_mat)
    regions = [region_for(label_mat_coo, label) for label in set(label_mat_coo.data)]
    return Many(regions)


def region_for(label_mat_coo, label):
    ind = label_mat_coo.data == label
    #TODO does this work in 3D?
    x = label_mat_coo.row[ind]
    y = label_mat_coo.col[ind]
    re = One(zip(x, y))
    return re
