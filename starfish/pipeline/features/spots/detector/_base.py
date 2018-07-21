from starfish.image import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class SpotFinderAlgorithmBase(AlgorithmBase):
    # TODO ambrosejcarr: should return IntensityTable when refactor is complete
    def find(self, hybridization_image: ImageStack):
        """Find spots."""
        raise NotImplementedError()
