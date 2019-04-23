from abc import abstractmethod
from typing import Callable, Sequence, Tuple, Type

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.pipeline.pipelinecomponent import PipelineComponent
from starfish.core.types import Number
from starfish.core.util import click
from starfish.core.util.click.indirectparams import CodebookParamType, ImageStackParamType
from .combine_adjacent_features import ConnectedComponentDecodingResult


class DetectPixels(PipelineComponent):
    """
    Decode an image by first coding each pixel, then combining the results into spots.
    """
    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        image_stack: ImageStack = ctx.obj["image_stack"]
        intensities: IntensityTable
        intensities, _ = instance.run(image_stack)
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
