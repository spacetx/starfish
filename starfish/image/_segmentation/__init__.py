from typing import Any, Dict, List, Type

import click
from skimage.io import imsave

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, PipelineComponent
from . import watershed
from ._base import SegmentationAlgorithmBase


class Segmentation(PipelineComponent):

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return SegmentationAlgorithmBase

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        pri_stack = ctx.obj["primary_images"]
        nuc_stack = ctx.obj["nuclei"]

        label_image = instance.run(pri_stack, nuc_stack)

        print(f"Writing label image to {output}")
        imsave(output, label_image)

    @staticmethod
    @click.group("segment")
    @click.option("--primary-images", required=True, type=click.Path(exists=True))
    @click.option("--nuclei", required=True, type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, primary_images, nuclei, output):
        print('Segmenting ...')
        ctx.obj = dict(
            component=Segmentation,
            output=output,
            primary_images=ImageStack.from_path_or_url(primary_images),
            nuclei=ImageStack.from_path_or_url(nuclei),
        )


Segmentation._cli_register()
