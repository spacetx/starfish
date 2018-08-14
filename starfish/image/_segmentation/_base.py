from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.stack import ImageStack


class SegmentationAlgorithmBase(AlgorithmBase):
    def run(self, hybridization_stack: ImageStack, nuclei_stack: ImageStack):
        """Performs registration on the stack provided."""
        raise NotImplementedError()
