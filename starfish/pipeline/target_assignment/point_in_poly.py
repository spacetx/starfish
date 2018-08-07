import numpy as np
import pandas as pd
import regional
from skimage.measure import points_in_poly
from tqdm import tqdm

from starfish.constants import Features
from starfish.intensity_table import IntensityTable
from ._base import TargetAssignmentAlgorithm


class PointInPoly2D(TargetAssignmentAlgorithm):

    def __init__(self, **kwargs) -> None:
        """
        PointInPoly accepts no parameters, but all pipeline components must accept arbitrary kwargs
        """

    @classmethod
    def add_arguments(cls, parser) -> None:
        pass

    @staticmethod
    def _assign(
            cells_region: regional.many, spots: pd.DataFrame, use_hull: bool=True, verbose:
            bool=False
    ) -> pd.DataFrame:

        results = pd.DataFrame({'spot_id': range(0, spots.shape[0])})
        results['cell_id'] = None

        if verbose:
            cell_iterator = tqdm(range(cells_region.count))
        else:
            cell_iterator = range(cells_region.count)

        for cell_id in cell_iterator:
            if use_hull:
                vertices = cells_region[cell_id].hull
            else:
                vertices = cells_region[cell_id].coordinates
            vertices = np.array(vertices)
            in_poly = points_in_poly(spots, vertices)
            results.loc[results.spot_id[in_poly], 'cell_id'] = cell_id

        return results

    def run(
            self, intensity_table: IntensityTable, regions: regional.many, verbose: bool=False
    ) -> pd.DataFrame:
        """Assign spots with target assignments to cells

        Parameters
        ----------
        intensity_table : IntensityTable
        regions : regional.many
            # TODO dganguli can you add an explanation? I'll fix during PR.
        verbose : bool
            If True, report on the progress of gene assignment (default False)

        Returns
        -------
        pd.DataFrame :
            DataFrame mapping of spot ids to cell ids
            # TODO should this be emitted as an extra column of IntensityTable, instead?

        """
        # TODO must support filtering on the passes filter column
        # TODO does this support 3d assignment?
        x = intensity_table.coords[Features.AXIS][Features.X].values
        y = intensity_table.coords[Features.AXIS][Features.Y].values
        points = pd.DataFrame(dict(x=x, y=y))
        return self._assign(regions, points, use_hull=True, verbose=verbose)
