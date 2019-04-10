from abc import abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

import numpy as np
import xarray as xr

from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.types import Axes, Number, SpotAttributes
from starfish.util import click
from starfish.util.click.indirectparams import ImageStackParamType

COMPONENT_NAME = "detect_spots"


class SpotFinder(PipelineComponent):
    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        image_stack = ctx.obj["image_stack"]
        blobs_stack = ctx.obj["blobs_stack"]
        blobs_axes = ctx.obj["blobs_axes"]

        intensities = instance.run(
            image_stack,
            blobs_stack,
            blobs_axes,
        )

        # When run() returns a tuple, we only save the intensities for now
        # TODO ambrosejcarr find a way to save arbitrary detector results
        if isinstance(intensities, tuple):
            intensities = intensities[0]
        intensities.save(output)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("-i", "--input", required=True, type=ImageStackParamType)
    @click.option("-o", "--output", required=True)
    @click.option(
        "--blobs-stack",
        default=None,
        required=False,
        type=ImageStackParamType,
        help="ImageStack that contains the blobs."
    )
    @click.option(
        "--blobs-axis",
        type=click.Choice([Axes.ROUND.value, Axes.CH.value, Axes.ZPLANE.value]),
        multiple=True,
        required=False,
        help="The axes that the blobs image will be maj-projected to produce the blobs_image"
    )
    @click.pass_context
    def _cli(ctx, input, output, blobs_stack, blobs_axis):
        """detect spots"""
        print('Detecting Spots ...')
        _blobs_axes = tuple(Axes(_blobs_axis) for _blobs_axis in blobs_axis)

        ctx.obj = dict(
            component=SpotFinder,
            image_stack=input,
            output=output,
            blobs_stack=blobs_stack,
            blobs_axes=_blobs_axes,
        )


class SpotFinderAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return SpotFinder

    @abstractmethod
    def run(
            self,
            primary_image: ImageStack,
            blobs_image: Optional[ImageStack] = None,
            blobs_axes: Optional[Tuple[Axes, ...]] = None,
            *args,
    ) -> Union[IntensityTable, Tuple[IntensityTable, Any]]:
        """Finds spots in an ImageStack"""
        raise NotImplementedError()

    @abstractmethod
    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:
        """Finds spots in a 3d volume"""
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
