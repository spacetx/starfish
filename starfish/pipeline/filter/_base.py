from typing import Optional

from starfish.image import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class FilterAlgorithmBase(AlgorithmBase):
    def filter(self, stack: ImageStack) -> Optional[ImageStack]:
        """Performs in-place filtering on an ImageStack."""
        raise NotImplementedError()
