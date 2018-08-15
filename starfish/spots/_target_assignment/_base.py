import pandas as pd
import regional

from starfish.intensity_table import IntensityTable
from starfish.pipeline.algorithmbase import AlgorithmBase


class TargetAssignmentAlgorithm(AlgorithmBase):
    def run(self, spots: IntensityTable, regions: regional.many) -> pd.DataFrame:
        """Performs target (e.g. gene) assignment given the spots and the regions."""
        raise NotImplementedError()
