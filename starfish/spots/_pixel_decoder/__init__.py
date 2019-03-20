from typing import Type

from starfish.codebook.codebook import Codebook
from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, import_all_submodules, PipelineComponent
from starfish.util import click
from . import _base
import_all_submodules(__file__, __package__)


COMPONENT_NAME = "detect_pixels"


class PixelSpotDecoder(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return _base.PixelDecoderAlgorithmBase

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        image_stack = ctx.obj["image_stack"]
        # TODO ambrosejcarr serialize and save ConnectedComponentDecodingResult somehow
        intensities, ccdr = instance.run(image_stack)
        intensities.save(output)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("-i", "--input", required=True, type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.option(
        '--codebook', default=None, required=True, help=(
            'A spaceTx spec-compliant json file that describes a three dimensional tensor '
            'whose values are the expected intensity of a spot for each code in each imaging '
            'round and each color channel.'
        )
    )
    @click.pass_context
    def _cli(ctx, input, output, codebook):
        """pixel-wise spot detection and decoding"""
        print('Detecting Spots ...')
        ctx.obj = dict(
            component=PixelSpotDecoder,
            image_stack=ImageStack.from_path_or_url(input),
            output=output,
            codebook=Codebook.from_json(codebook),
        )
