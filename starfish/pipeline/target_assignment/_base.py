from starfish.pipeline.algorithmbase import AlgorithmBase


class TargetAssignmentAlgorithm(AlgorithmBase):
    def assign_targets(self, spots, regions):
        """Performs target (e.g. gene) assignment given the spots and the regions."""
        raise NotImplementedError()
