import numpy as np
import pandas as pd
from functools import reduce
import scipy.ndimage.measurements as spm
from regional import many as Many
from regional import one as One
from scipy.sparse import coo_matrix


def stack_describe(stack):
    num_rounds = stack.shape[0]
    stats = [im_describe(stack[k, :]) for k in range(num_rounds)]
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
        # LOL -- in python3 zip returns an iterator. to force it to
        # a list we call list(zip). In Python 2.7 this is effectively a noop
        re = One(list(zip(x, y)))
        return re

    unique_labels = sorted(set(label_mat_coo.data))
    regions = [region_for(label_mat_coo, label) for label in unique_labels]

    return Many(regions)


def measure(im, labels, num_objs, measurement_type='mean'):
    if measurement_type == 'mean':
        res = spm.mean(im, labels, range(1, num_objs))
    elif measurement_type == 'max':
        res = spm.maximum(im, labels, range(1, num_objs))
    else:
        raise ValueError('Unsporrted measurement type: {}'.format(measurement_type))

    return res


def measure_stack(stack, labels, num_objs, measurement_type='mean'):
    from starfish.munge import stack_to_list
    ims = stack_to_list(stack)
    res = [measure(im, labels, num_objs, measurement_type) for im in ims]
    return res
