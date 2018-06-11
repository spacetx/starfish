import pandas as pd

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

    def assign_genes(self, intensity_table, regions):
        from starfish.assign import assign

        x = intensity_table.coords['features'].x.values
        y = intensity_table.coords['features'].y.values
        points = pd.DataFrame(dict(x=x, y=y))
        return assign(regions, points, use_hull=True)
