import numpy as np
import xarray as xr
from functools import partial
from typing import List, Union, Mapping, Optional, Tuple

from skimage import transform
from skimage.transform._geometric import GeometricTransform

from starfish.imagestack.imagestack import ImageStack
from starfish.util import click
from starfish.image._apply_transform._base import ApplyTransformBase
from starfish.types import Axes


class Warp(ApplyTransformBase):

    @staticmethod
    def _warp(image: Union[xr.DataArray, np.ndarray], transformation_object: GeometricTransform
              ) -> np.ndarray:
        """ Wrapper around skimage.transform.warp. Warps an image according to a
        given coordinate transformation.
        image: the image to be transformed
        transformation_object: skimage.transform.GeometricTransform
        Returns
        -------
        """
        return transform.warp(image, transformation_object)

    def run(
            self, stack: ImageStack,
            transforms_list: List[Tuple[Mapping[Axes, int], GeometricTransform]],
            in_place: bool=False, verbose: bool=False,
            n_processes: Optional[int]=None) -> ImageStack:
        """Applies a transformation to an Imagestack
        Parameters
        ----------
        stack : ImageStack
            Stack to be transformed.
        transforms_list:
            TALJBGWOUGBOUWR:BWOURGB:WURGB
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
        all_axes = {Axes.ROUND, Axes.CH, Axes.ZPLANE}
        for selector, transformation_object in transforms_list:
            other_axes = all_axes - {list(selector.keys())[0]}
            for axes in stack._iter_axes(other_axes):
                selector.update(axes)
                selected_image = np.squeeze(stack.sel(selector).xarray)
                transformed_image = Warp._warp(selected_image, transformation_object)
                stack.set_slice(selector, transformed_image.astype(np.float32))

        return stack

    @staticmethod
    @click.command("Warp")
    @click.pass_context
    def _cli(ctx):
        ctx.obj["component"]._cli_run(ctx, Warp())
