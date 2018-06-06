from starfish.pipeline.algorithmbase import AlgorithmBase


class PixelFinderAlgorithmBase(AlgorithmBase):
    def find(self, stack):
        """Find spots."""
        raise NotImplementedError()
