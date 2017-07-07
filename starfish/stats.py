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


def label_to_regions(labels):
    label_mat_coo = coo_matrix(labels)

    def region_for(label_mat_coo, label):
        ind = label_mat_coo.data == label
        # TODO does this work in 3D?
        x = label_mat_coo.row[ind]
        y = label_mat_coo.col[ind]
        re = One(zip(x, y))
        return re

    regions = [region_for(label_mat_coo, label) for label in set(label_mat_coo.data)]

    return Many(regions)


def measure_mean(im, labels, num_objs):
    means = spm.mean(im, labels, range(0, num_objs))
    return means


def measure_mean_stack(stack, labels, num_objs):
    ims = stack_to_list(stack)
    res = [measure_mean(im, labels, num_objs) for im in ims]
    return res
