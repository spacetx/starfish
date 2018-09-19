from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class SegmentationAlgorithmBase(AlgorithmBase):
    def run(self, hybridization_stack: ImageStack, nuclei_stack: ImageStack):
        """Performs registration on the stack provided."""
        raise NotImplementedError()
