import numpy as np
import pandas as pd
from skimage.measure import points_in_poly

from .stats import label_to_regions


def assign(cells_label, spots, use_hull=True, verbose=False):
    cells_region = label_to_regions(cells_label)

    res = pd.DataFrame({'spot_id': range(0, spots.shape[0])})
    res['cell_id'] = None

    for cell_id in range(cells_region.count):
        if use_hull:
            verts = cells_region[cell_id].hull
        else:
            verts = cells_region[cell_id].coordinates
        verts = np.array(verts)
        in_poly = points_in_poly(spots, verts)
        res.loc[res.spot_id[in_poly], 'cell_id'] = cell_id
        if verbose:
            cnt = np.sum(in_poly)
            print(cell_id, cnt)

    return res
