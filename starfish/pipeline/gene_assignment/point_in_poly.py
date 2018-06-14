import numpy as np
import pandas as pd
from skimage.measure import points_in_poly

from ._base import GeneAssignmentAlgorithm


class PointInPoly(GeneAssignmentAlgorithm):
    def __init__(self, **kwargs):
        pass

    @classmethod
    def get_algorithm_name(cls):
        return "point_in_poly"

    @classmethod
    def add_arguments(cls, parser):
        pass

    def assign_genes(self, spots, regions):
        # TODO only works in 3D
        points = spots.loc[:, ['x', 'y']].values
        return self._assign(regions, points, use_hull=True)

    @staticmethod
    def _assign(cells_region, spots, use_hull=True, verbose=False):
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
