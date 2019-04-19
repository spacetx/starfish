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


class DetectPixels(PipelineComponent):
    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        image_stack: ImageStack = ctx.obj["image_stack"]
        # TODO ambrosejcarr serialize and save ConnectedComponentDecodingResult somehow

        intensities: IntensityTable
        ccdr: ConnectedComponentDecodingResult
        intensities, ccdr = instance.run(image_stack)
        intensities.to_netcdf(output)

    @staticmethod
    @click.group("DetectPixels")
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
            component=DetectPixels,
            image_stack=input,
            output=output,
            codebook=codebook,
        )


class DetectPixelsAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return DetectPixels

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
