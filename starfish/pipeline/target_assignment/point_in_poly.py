import numpy as np
from skimage.measure import points_in_poly
import pandas as pd

from starfish.constants import Features
from ._base import TargetAssignmentAlgorithm


class PointInPoly(TargetAssignmentAlgorithm):
    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose

    @classmethod
    def add_arguments(cls, parser):
        pass

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

    def assign_targets(self, intensity_table, regions):

        x = intensity_table.coords[Features.AXIS][Features.X].values
        y = intensity_table.coords[Features.AXIS][Features.Y].values
        points = pd.DataFrame(dict(x=x, y=y))
        return self._assign(regions, points, use_hull=True, verbose=self.verbose)
