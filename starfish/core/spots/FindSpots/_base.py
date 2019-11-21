from abc import abstractmethod
from typing import Callable, Optional

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.types import Number, SpotFindingResults


class FindSpotsAlgorithm(metaclass=AlgorithmBase):
    """
    Starfish spot finders use a variety of means to detect bright spots against
    dark backgrounds. Starfish's spot detectors each have different strengths and weaknesses.

    **Fixed-position spot finders**

    The following spot finders have two modes of operation.

    The first mode is suitable for coded
    experiments where genes are identified by patterns of spots over all rounds and channels of the
    experiment. In this mode, the spot finders identify spots in a single reference image,
    which can be either a dots auxiliary image, or a maximum intensity projection of the primary
    images. The positions of the maxima are then measured in all other images, and the intensities
    across the complete experiment are stored in an :ref:`IntensityTable`

    The second mode is suitable for assays that detect spots in a single round, such as single
    molecule FISH and RNAscope. This mode simply finds all the spots and concatenates them into a
    long-form IntensityTable. In this mode, the spots are not measured in images that correspond to
    other :code:`(round, channel)` pairs; those positions of the IntensityTable are filled with
    :code:`np.nan`.

    1. The :py:class:`~starfish.spots._find_spots.blob.BlobDetector` allows the user to pre-filter
    an image using either a Laplacian-of-Gaussians or
    Difference-of-Gaussians (fast approximation to Laplacian-of-Gaussians). These filters are
    applied at with a user-specified variety of Gaussian kernel sizes, and the best-fitting size is
    automatically selected. This allows this filter to detect Gaussian shaped blobs of various
    sizes.

    """

    @abstractmethod
    def run(self, image_stack: ImageStack,
            reference_image: Optional[ImageStack] = None, *args) -> SpotFindingResults:
        """Find and measure spots across rounds and channels in the provided ImageStack."""
        raise NotImplementedError()

    @staticmethod
    def _get_measurement_function(
            measurement_type: str
    ) -> Callable[[np.ndarray], Number]:
        try:
            measurement_function = getattr(np, measurement_type)
        except AttributeError:
            raise ValueError(
                f'measurement_type must be a numpy reduce function such as "max" or "mean". '
                f'{measurement_type} not found.')
        return measurement_function
