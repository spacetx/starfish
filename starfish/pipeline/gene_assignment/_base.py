from starfish.pipeline.algorithmbase import AlgorithmBase


class GeneAssignmentAlgorithm(AlgorithmBase):
    def assign_genes(self, spots, regions):
        """Performs gene assignment given the spots and the regions."""
        raise NotImplementedError()
