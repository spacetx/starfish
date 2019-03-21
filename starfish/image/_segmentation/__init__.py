from typing import Type

from skimage.io import imsave

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, import_all_submodules, PipelineComponent
from starfish.util import click
from . import _base
import_all_submodules(__file__, __package__)


COMPONENT_NAME = "segment"


class Segmentation(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return _base.SegmentationAlgorithmBase

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        pri_stack = ctx.obj["primary_images"]
        nuc_stack = ctx.obj["nuclei"]

        label_image = instance.run(pri_stack, nuc_stack)

        print(f"Writing label image to {output}")
        imsave(output, label_image)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("--primary-images", required=True, type=click.Path(exists=True))
    @click.option("--nuclei", required=True, type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, primary_images, nuclei, output):
        """define polygons for cell boundaries and assign spots"""
        print('Segmenting ...')
        ctx.obj = dict(
            component=Segmentation,
            output=output,
            primary_images=ImageStack.from_path_or_url(primary_images),
            nuclei=ImageStack.from_path_or_url(nuclei),
        )
