import numpy as np
from skimage.feature import register_translation
from skimage.transform._geometric import SimilarityTransform

from starfish.core.image._registration.transforms_list import TransformsList
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, TransformType
from ._base import LearnTransformAlgorithm


class Translation(LearnTransformAlgorithm):
    """
    Iterate over the given axes of an ImageStack and learn the translation transform
    based off the reference_stack passed into :py:class:`Translation`'s constructor.
    Only supports 2d data.

    Parameters
    ----------
    axes : Axes
        The axes {r, ch, zplane} to iterate over
    reference_stack : ImageStack
        The target image used in :py:func:`skimage.feature.register_translation`
    upsampling : int
        upsampling factor (default=1). See :py:func:`~skimage.registration.phase_cross_correlation`
        for an explanation of this parameter. In brief, this parameter determines the resolution of
        the registration. A value of 1 represents pixel resolution, a value of 10 is 1/10th of
        a pixel, a value of 300 is 1/300th of a pixel, and so on.
    """

    def __init__(self, reference_stack: ImageStack, axes: Axes, upsampling: int=1):

        self.upsampling = upsampling
        self.axes = axes
        self.reference_stack = reference_stack

    def run(self, stack: ImageStack, verbose: bool=False, *args) -> TransformsList:
        """
        Iterate over the given axes of an ImageStack and learn the translation transform
        based off the reference_stack passed into :py:class:`Translation`'s constructor.
        Only supports 2d data.

        Parameters
        ----------
        stack : ImageStack
            Stack to calculate the transforms on.
        verbose : bool
            if True, report on transformation progress (default = False)

        Returns
        -------
        List[Tuple[Mapping[Axes, int], SimilarityTransform]] :
            A list of tuples containing axes of the Imagestack and associated
            transform to apply.
        """

        transforms = TransformsList()
        reference_image = np.squeeze(self.reference_stack.xarray)
        for a in stack.axis_labels(self.axes):
            target_image = np.squeeze(stack.sel({self.axes: a}).xarray)
            if len(target_image.shape) != 2:
                raise ValueError(
                    f"Only axes: {self.axes.value} can have a length > 1, "
                    f"please use the MaxProj filter."
                )

            shift, error, phasediff = register_translation(src_image=target_image,
                                                           target_image=reference_image,
                                                           upsample_factor=self.upsampling)
            if verbose:
                print(f"For {self.axes}: {a}, Shift: {shift}, Error: {error}")
            selectors = {self.axes: a}
            # reverse shift because SimilarityTransform stores in y,x format
            shift = shift[::-1]
            transforms.append(selectors,
                              TransformType.SIMILARITY,
                              SimilarityTransform(translation=shift))

        return transforms
