from abc import abstractmethod
from typing import Any, Callable, Sequence, Tuple, Type, Union

import click
import numpy as np
import xarray as xr

from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.types import Axes, Number, SpotAttributes

COMPONENT_NAME = "detect_spots"


class SpotFinder(PipelineComponent):
    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        blobs_stack = ctx.obj["blobs_stack"]
        image_stack = ctx.obj["image_stack"]
        ref_image = ctx.obj["reference_image_from_max_projection"]
        if blobs_stack is not None:
            blobs_stack = ImageStack.from_path_or_url(blobs_stack)  # type: ignore
            mp = blobs_stack.max_proj(Axes.ROUND, Axes.CH)
            mp_numpy = mp._squeezed_numpy(Axes.ROUND, Axes.CH)
            intensities = instance.run(
                image_stack,
                blobs_image=mp_numpy,
                reference_image_from_max_projection=ref_image,
            )
        else:
            intensities = instance.run(image_stack)

        # When run() returns a tuple, we only save the intensities for now
        # TODO ambrosejcarr find a way to save arbitrary detector results
        if isinstance(intensities, tuple):
            intensities = intensities[0]
        intensities.save(output)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("-i", "--input", required=True, type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.option(
        '--blobs-stack', default=None, required=False, help=(
            'ImageStack that contains the blobs. Will be max-projected across imaging round '
            'and channel to produce the blobs_image'
        )
    )
    @click.option(
        '--reference-image-from-max-projection', default=False, is_flag=True, help=(
            'Construct a reference image by max projecting imaging rounds and channels. Spots '
            'are found in this image and then measured across all images in the input stack.'
        )
    )
    @click.pass_context
    def _cli(ctx, input, output, blobs_stack, reference_image_from_max_projection):
        """detect spots"""
        print('Detecting Spots ...')
        ctx.obj = dict(
            component=SpotFinder,
            image_stack=ImageStack.from_path_or_url(input),
            output=output,
            blobs_stack=blobs_stack,
            reference_image_from_max_projection=reference_image_from_max_projection,
        )


class SpotFinderAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return SpotFinder

    @abstractmethod
    def run(
            self,
            primary_image: ImageStack,
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
