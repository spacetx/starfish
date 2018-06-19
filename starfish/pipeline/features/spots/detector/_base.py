from typing import Tuple

from starfish.image import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.features.spot_attributes import SpotAttributes
from starfish.pipeline.features.encoded_spots import EncodedSpots


class SpotFinderAlgorithmBase(AlgorithmBase):
    def find(self, hybridization_image: ImageStack) -> Tuple[SpotAttributes, EncodedSpots]:
        """Find spots."""
        raise NotImplementedError()
