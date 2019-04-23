from abc import abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

import numpy as np
import xarray as xr

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.pipeline.pipelinecomponent import PipelineComponent
from starfish.core.types import Axes, Number, SpotAttributes
from starfish.core.util import click
from starfish.core.util.click.indirectparams import ImageStackParamType


class DetectSpots(PipelineComponent):
    """
    Starfish spot detectors use a variety of means to detect bright spots against
    dark backgrounds. Starfish's spot detectors each have different strengths and weaknesses.

    **Fixed-position spot finders**

    The following spot finders have two modes of operation.

    The first mode is suitable for coded
    experiments where genes are identified by patterns of spots over all rounds and channels of the
    experiment. In this mode, the spot finders identify spots in a single image, which can be either
    a dots auxiliary image, or a maximum intensity projection of the primary images.
    The positions of the maxima are then measured in all other images, and the intensities across
    the complete experiment are stored in an :ref:`IntensityTable`

    The second mode is suitable for assays that detect spots in a single round, such as single
    molecule FISH and RNAscope. This mode simply finds all the spots and concatenates them into a
    long-form IntensityTable. In this mode, the spots are not measured in images that correspond to
    other :code:`(round, channel)` pairs; those positions of the IntensityTable are filled with
    :code:`np.nan`.

    1. The :py:class:`~starfish.spots._detect_spots.blob.BlobDetector` allows the user to pre-filter
    an image using either a Laplacian-of-Gaussians or
    Difference-of-Gaussians (fast approximation to Laplacian-of-Gaussians). These filters are
    applied at with a user-specified variety of Gaussian kernel sizes, and the best-fitting size is
    automatically selected. This allows this filter to detect Gaussian shaped blobs of various
    sizes.

    2. The :py:class:`~starfish.spots._detect_spots.local_max_peak_finder.LocalMaxPeakFinder`
    identifies local maxima using the same machinery as the BlobDetector, except that
    it requires that the user to pre-apply any filters to enhance spots. In exchange, it allows a
    user to automatically select the threshold that separates foreground (spots) from background
    (noise).

    In the future, starfish will combine the functionality of LocalMaxPeakFinder into BlobDetector
    so that a user can detect blobs of multiple sizes *and* automatically find a stable threshold.

    3. The
    :py:class:`~starfish.spots._detect_spots.trackpy_local_max_peak_finder.TrackpyLocalMaxPeakFinder`
    provides an implementation of the `Crocker-Grier <crocker_grier>`_ spot finding
    algorithm. This method optionally preprocesses the image by performing a band pass and a
    threshold. It then locates all peaks of brightness, characterizes the neighborhoods of the peaks
    and takes only those with given total brightness (“mass”). Finally, it refines the positions of
    each peak.

    .. _crocker_grier: https://physics.nyu.edu/grierlab/methods3c/

    **Fuzzy-position spot finders**

    In addition to the spot finders above, we expose single additional spot finder that, while very
    similar in implementation to the BlobFinder, is able to adjust to small local spatial
    perturbations in the centroid of the spot across rounds and channels, such as those that might
    occur from small shifts to the tissue or stage.

    1. The
    :py:class:`~starfish.spots._detect_spots.local_search_blob_detector.LocalSearchBlobDetector`
    is a Gaussian blob detector finds spots in all rounds and channels independently, then, given
    each spot in a user specified "anchor round", selects the closest spot by spatial position in
    all other rounds and aggregates those into codes which can subsequently be decoded. This Spot
    detector is only applicable to experiments with "one-hot" codebooks, such as those generated
    by in-situ sequencing, which guarantee that only one channel will be "on" per round.

    """
    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        image_stack = ctx.obj["image_stack"]
        blobs_stack = ctx.obj["blobs_stack"]
        blobs_axes = ctx.obj["blobs_axes"]

        intensities: IntensityTable = instance.run(
            image_stack,
            blobs_stack,
            blobs_axes,
        )

        # TODO ambrosejcarr find a way to save arbitrary detector results
        intensities.to_netcdf(output)

    @staticmethod
    @click.group("DetectSpots")
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
            component=DetectSpots,
            image_stack=input,
            output=output,
            blobs_stack=blobs_stack,
            blobs_axes=_blobs_axes,
        )


class DetectSpotsAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return DetectSpots

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
