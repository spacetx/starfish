from typing import Union

import numpy as np
from skimage.feature import register_translation
from skimage.transform._geometric import SimilarityTransform

from starfish.image._learn_transform.transforms_list import TransformsList
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes, TransformType
from starfish.util import click
from ._base import LearnTransformBase


class Translation(LearnTransformBase):

    def __init__(self, reference_stack: Union[str, ImageStack], axis: Axes, upsampling: int=1):
        """
        Parameters
        ----------
        axis:
            The aixs {r, ch, zplane} to iterate over
        reference_stack: ImageStack
            The target image used in skimage.feature.register_translation
        upsampling: int
            upsampling factor
        """
        self.upsampling = upsampling
        self.axis = axis
        if isinstance(reference_stack, ImageStack):
            self.reference_stack = reference_stack
        else:
            self.reference_stack = ImageStack.from_path_or_url(reference_stack)

    def run(self, stack: ImageStack) -> TransformsList:
        """
        Iterate over the given axis of an ImageStack and learn the Similarity transform
        based off the instantiated reference_image.

        Parameters
        ----------
        stack : ImageStack
            Stack to calculate the transforms on.

        Returns
        -------
        List[Tuple[Any, SimilarityTransform]] :
            A list of tuples containing axes of the Imagestack and associated
            transform to apply.
        """

        transforms = TransformsList()
        reference_image = np.squeeze(self.reference_stack.xarray)
        for a in stack.axis_labels(self.axis):
            target_image = np.squeeze(stack.sel({self.axis: a}).xarray)
            if len(target_image.shape) != 2:
                raise ValueError(
                    "Only axes: " + self.axis.value + " can have a length > 1, "
                                                      "please us the MaxProj filter."
                )

            shift, error, phasediff = register_translation(src_image=target_image,
                                                           target_image=reference_image,
                                                           upsample_factor=self.upsampling)
            print(f"For {self.axis}: {a}, Shift: {shift}, Error: {error}")
            selectors = {self.axis: a}
            # reverse shift because SimilarityTransform stores in y,x format
            shift = shift[::-1]
            transforms.append(selectors,
                              TransformType.SIMILARITY,
                              SimilarityTransform(translation=shift))

        return transforms

    @staticmethod
    @click.command("Translation")
    @click.option("--reference-stack", required=True, type=click.Path(exists=True),
                  help="The image to align the input ImageStack to.")
    @click.option("--axis", default="r", type=str, help="The axis to iterate over.")
    @click.option("--upsampling", default=1, type=int, help="Upsampling factor.")
    @click.pass_context
    def _cli(ctx, reference_stack, axis, upsampling):
        ctx.obj["component"]._cli_run(ctx, Translation(
            reference_stack=reference_stack, axis=Axes(axis), upsampling=upsampling))
