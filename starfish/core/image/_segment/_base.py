from abc import abstractmethod
from typing import Type

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.pipeline import PipelineComponent
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.segmentation_mask import SegmentationMaskCollection
from starfish.core.util import click
from starfish.core.util.click.indirectparams import ImageStackParamType


class Segment(PipelineComponent):
    """
    Starfish class implementing segmentation approaches.
    """
    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        pri_stack = ctx.obj["primary_images"]
        nuc_stack = ctx.obj["nuclei"]

        masks = instance.run(pri_stack, nuc_stack)

        print(f"Writing masks to {output}")
        masks.save(output)

    @staticmethod
    @click.group("Segment")
    @click.option("--primary-images", required=True, type=ImageStackParamType)
    @click.option("--nuclei", required=True, type=ImageStackParamType)
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, primary_images, nuclei, output):
        """define masks for cell boundaries and assign spots"""
        print('Segmenting ...')
        ctx.obj = dict(
            component=Segment,
            output=output,
            primary_images=primary_images,
            nuclei=nuclei,
        )


class SegmentAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return Segment

    @abstractmethod
    def run(
            self,
            primary_image_stack: ImageStack,
            nuclei_stack: ImageStack,
            *args
    ) -> SegmentationMaskCollection:
        """Performs segmentation on the stack provided."""
        raise NotImplementedError()
