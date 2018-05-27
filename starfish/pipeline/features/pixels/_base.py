from starfish.pipeline.algorithmbase import AlgorithmBase


class PixelFinderAlgorithmBase(AlgorithmBase):
    def find(self, stack):  # -> Tuple[SpotAttributes, EncodedSpots]:
        """Find spots."""
        raise NotImplementedError()
