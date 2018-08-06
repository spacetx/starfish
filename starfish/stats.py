from functools import reduce
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import scipy.ndimage.measurements as spm
import regional
from scipy.sparse import coo_matrix

from starfish.image import ImageStack
from starfish.types import Number


def stack_describe(stack: np.ndarray) -> List[Dict[str, Number]]:
    num_rounds = stack.shape[0]
    stats = [im_describe(stack[k, :]) for k in range(num_rounds)]
    return stats


def im_describe(im: np.ndarray) -> Dict[str, Number]:
    shape = im.shape
    flat_dims = reduce(lambda x, y: x * y, shape)
    flat_im = np.reshape(im, flat_dims)
    stats = pd.Series(flat_im).describe()
    return stats.to_dict()


def label_to_regions(labels) -> regional.many:
    label_mat_coo = coo_matrix(labels)

    def region_for(label_mat_coo, label):
        ind = label_mat_coo.data == label
        # TODO does this work in 3D?
        x = label_mat_coo.row[ind]
        y = label_mat_coo.col[ind]

        re = regional.one(list(zip(x, y)))
        return re

    unique_labels = sorted(set(label_mat_coo.data))
    regions = [region_for(label_mat_coo, label) for label in unique_labels]

    return regional.many(regions)


def measure(
        im: np.ndarray, labels: Sequence[Number], num_objs: int, measurement_type: str='mean'
) -> List[float]:
    if measurement_type == 'mean':
        res = spm.mean(im, labels, range(1, num_objs))
    elif measurement_type == 'max':
        res = spm.maximum(im, labels, range(1, num_objs))
    else:
        raise ValueError(f'Unsupported measurement type: {measurement_type}. Choose one of "mean" '
                         f'or "max".')

    return res


def measure_stack(
        stack: np.ndarray, labels: Sequence[Number], num_objs: int, measurement_type='mean'
) -> List[List[float]]:
    from starfish.munge import stack_to_list
    ims = stack_to_list(stack)
    res = [measure(im, labels, num_objs, measurement_type) for im in ims]
    return res
