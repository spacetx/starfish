import numpy as np
import xarray as xr
from functools import partial
from typing import Union, Optional

from skimage import transform
from skimage.transform._geometric import GeometricTransform

from starfish.imagestack.imagestack import ImageStack
from starfish.util import click
from starfish.image._apply_transform._base import ApplyTransformBase
from starfish.types import Axes


class Warp(ApplyTransformBase):

    def __init__(self, transformation_object: GeometricTransform):
        self.transformation_object = transformation_object

    @staticmethod
    def _warp(image: Union[xr.DataArray, np.ndarray], transformation_object: GeometricTransform
              ) -> np.ndarray:
        """ Wrapper around skimage.transform.warp. Warps an image according to a
        given coordinate transformation.
        image:
        transformation_object:
        Returns
        -------
        """
        return transform.warp(image, transformation_object)

    def run(
            self, stack: ImageStack, in_place: bool=False, verbose: bool=False,
            n_processes: Optional[int]=None) -> ImageStack:
        """Applies a transformation to an Imagestack
        Parameters
        ----------
        stack : ImageStack
            Stack to be transformed.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on filtering progress (default = False)
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter
        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.
        """




        group_by = {Axes.CH, Axes.ROUND, Axes.ZPLANE}
        warp = partial(self._warp, transformation_object=self.transformation_object)
        result = stack.apply(
            warp,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )

        return result

    @staticmethod
    @click.command("Warp")
    @click.option(
        "--transform-object", default=None, type=int, help="clip intensities below this percentile")
    @click.pass_context
    def _cli(ctx, transform_object):
        ctx.obj["component"]._cli_run(ctx, Warp(transformation_object=transform_object))
