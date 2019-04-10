from abc import abstractmethod
from typing import Callable, Sequence, Tuple, Type

import numpy as np

from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.types import Number
from starfish.util import click
from starfish.util.click.indirectparams import CodebookParamType, ImageStackParamType
from .combine_adjacent_features import ConnectedComponentDecodingResult


COMPONENT_NAME = "detect_pixels"


class PixelSpotDecoder(PipelineComponent):
    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        image_stack = ctx.obj["image_stack"]
        # TODO ambrosejcarr serialize and save ConnectedComponentDecodingResult somehow
        intensities, ccdr = instance.run(image_stack)
        intensities.save(output)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("-i", "--input", required=True, type=ImageStackParamType)
    @click.option("-o", "--output", required=True)
    @click.option(
        "--codebook",
        default=None, required=True, type=CodebookParamType,
        help=(
            "A spaceTx spec-compliant json file that describes a three dimensional tensor "
            "whose values are the expected intensity of a spot for each code in each imaging "
            "round and each color channel."
        )
    )
    @click.pass_context
    def _cli(ctx, input, output, codebook):
        """pixel-wise spot detection and decoding"""
        print('Detecting Spots ...')
        ctx.obj = dict(
            component=PixelSpotDecoder,
            image_stack=input,
            output=output,
            codebook=codebook,
        )


class PixelDecoderAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return PixelSpotDecoder

    @abstractmethod
    def run(
            self,
            primary_image: ImageStack,
            *args,
    ) -> Tuple[IntensityTable, ConnectedComponentDecodingResult]:
        """Finds spots in an ImageStack"""
        raise NotImplementedError()

    @staticmethod
    def _get_measurement_function(measurement_type: str) -> Callable[[Sequence], Number]:
        try:
            measurement_function = getattr(np, measurement_type)
        except AttributeError:
            raise ValueError(
                f'measurement_type must be a numpy reduce function such as "max" or "mean". '
                f'{measurement_type} not found.')
        return measurement_function
