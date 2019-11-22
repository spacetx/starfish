from abc import abstractmethod

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.pipeline.algorithmbase import AlgorithmBase


class FilterAlgorithm(metaclass=AlgorithmBase):

    @abstractmethod
    def run(
            self,
            binary_mask_collection: BinaryMaskCollection,
            *args,
            **kwargs
    ) -> BinaryMaskCollection:
        """Performs a filter on the binary mask collection provided."""
        raise NotImplementedError()
