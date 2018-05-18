from starfish.pipeline.algorithmbase import AlgorithmBase


class SegmentationAlgorithmBase(AlgorithmBase):
    def segment(self, stack):
        """Performs registration on the stack provided."""
        raise NotImplementedError()
