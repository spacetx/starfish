import numpy as np
import pandas as pd
import regional
from skimage.measure import points_in_poly
from tqdm import tqdm

from starfish.intensity_table import IntensityTable
from starfish.types import Features, Indices
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
            cells_region: regional.many, intensities: IntensityTable, use_hull: bool=True, verbose:
            bool=False
    ) -> IntensityTable:

        x = intensities.coords[Indices.X.value].values
        y = intensities.coords[Indices.Y.value].values
        points = pd.DataFrame(dict(x=x, y=y))

        results = pd.DataFrame({'spot_id': range(0, intensities.shape[0])})
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
            in_poly = points_in_poly(points, vertices)
            results.loc[results.spot_id[in_poly], 'cell_id'] = cell_id

        intensities['cell_id'] = (Features.AXIS, results['cell_id'])

        return intensities

    def run(
            self, intensity_table: IntensityTable, regions: regional.many, verbose: bool=False
    ) -> IntensityTable:
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
        IntensityTable :
            IntensityTable with added features variable containing cell ids

        """
        # TODO must support filtering on the passes filter column
        return self._assign(regions, intensity_table, use_hull=True, verbose=verbose)
