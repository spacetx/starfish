from typing import List, Mapping, Tuple

from skimage.transform._geometric import GeometricTransform

from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.types import Axes


class LearnTransformBase(AlgorithmBase):
    def run(self, stack, axis) -> List[Tuple[Mapping[Axes, int], GeometricTransform]]:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
