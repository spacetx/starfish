from typing import Optional

from starfish.image import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class FilterAlgorithmBase(AlgorithmBase):
    def run(self, stack: ImageStack) -> Optional[ImageStack]:
        """Performs filtering on an ImageStack."""
        raise NotImplementedError()
