from ._base import GeneAssignmentAlgorithm


class PointInPoly(GeneAssignmentAlgorithm):
    @classmethod
    def from_cli_args(cls, args):
        return PointInPoly()

    @classmethod
    def get_algorithm_name(cls):
        return "point_in_poly"

    @classmethod
    def add_arguments(cls, parser):
        pass

    def assign_genes(self, spots, regions):
        from starfish.assign import assign

        # TODO only works in 3D
        points = spots.loc[:, ['x', 'y']].values
        return assign(regions, points, use_hull=True)
