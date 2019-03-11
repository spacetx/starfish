import numpy as np
from typing import Any, List, Tuple, Union

from skimage.feature import register_translation
from skimage.transform._geometric import SimilarityTransform

from starfish.imagestack.imagestack import ImageStack
from starfish.image._registration._base import RegistrationAlgorithmBase
from starfish.types import Axes
from starfish.util import click


class Translation(RegistrationAlgorithmBase):

    def __init__(self, reference_stack: Union[str, ImageStack], upsampling: int=1):
        self.upsampling = upsampling
        if isinstance(reference_stack, ImageStack):
            self.reference_stack = reference_stack
        else:
            self.reference_stack = ImageStack.from_path_or_url(reference_stack)

    def run(self, stack: ImageStack) -> List[Tuple[Any, SimilarityTransform]]:
        transforms: List[Tuple[Any, SimilarityTransform]] = list()
        reference_image = np.squeeze(self.reference_stack.xarray)
        for r in stack.axis_labels(Axes.ROUND):
            selectecd_round = stack.sel({Axes.ROUND: r}).xarray
            target_image = selectecd_round.max([Axes.ROUND, Axes.CH.value, Axes.ZPLANE.value])
            shift, error, phasediff = register_translation(target_image,
                                                           reference_image,
                                                           upsample_factor=self.upsampling)
            selectors = {Axes.ROUND: r}
            transforms.append((selectors, SimilarityTransform(translation=shift)))

        return transforms

    @staticmethod
    @click.command("Translation")
    @click.option("--reference-stack", required=True, type=click.Path(exists=True),
                  help="The image stack to align the input image stack to.")
    @click.pass_context
    def _cli(ctx, reference_stack):
        ctx.obj["component"]._cli_run(ctx, Translation(reference_stack=reference_stack))

