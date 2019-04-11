from abc import abstractmethod
from typing import Type

from skimage.io import imsave

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import PipelineComponent
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.util import click
from starfish.util.click.indirectparams import ImageStackParamType

COMPONENT_NAME = "segment"


class Segmentation(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

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
    @click.option("--primary-images", required=True, type=ImageStackParamType)
    @click.option("--nuclei", required=True, type=ImageStackParamType)
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, primary_images, nuclei, output):
        """define polygons for cell boundaries and assign spots"""
        print('Segmenting ...')
        ctx.obj = dict(
            component=Segmentation,
            output=output,
            primary_images=primary_images,
            nuclei=nuclei,
        )


class SegmentationAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return Segmentation

    @abstractmethod
    def run(self, primary_image_stack: ImageStack, nuclei_stack: ImageStack, *args):
        """Performs registration on the stack provided."""
        raise NotImplementedError()
