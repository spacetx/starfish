from starfish.pipeline.algorithmbase import AlgorithmBase


class SegmentationAlgorithmBase(AlgorithmBase):
    def segment(self, hybridization_stack, nuclei_stack):
        """Performs registration on the stack provided."""
        raise NotImplementedError()
