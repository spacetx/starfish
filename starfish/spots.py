import scipy.ndimage.measurements as spm
import numpy as np
import pandas as pd

from starfish.filters import bin_thresh
from starfish.munge import scale, max_proj, gather
from starfish.stats import label_to_regions, measure_mean_stack


class SimpleSpotDetector:
    def __init__(self, stack, thresh):
        self.stack = stack
        self.thresh = thresh

    def go(self):
        mp = scale(self.stack, 'max')
        mp = max_proj(mp)
        mp_thresh = bin_thresh(mp,self.thresh)
        labels, num_objs = spm.label(mp_thresh)

        self.areas = spm.sum(np.ones(labels.shape),
                             labels,
                             np.array(range(0, num_objs + 1), dtype=np.int32))

        e = self.areas[(self.areas > 1) & (self.areas <= 10)]

        self.regions = label_to_regions(labels)

        res = measure_mean_stack(self.stack, labels, num_objs)

        num_hybs = self.stack.shape[0]
        cols = range(num_hybs)
        cols = ['hyb_{}'.format(c + 1) for c in cols]
        d = dict(zip(cols, res))
        d['spot_id'] = range(num_objs)

        t = pd.DataFrame(d)
        f = gather(t, 'hybs', 'vals', cols)
