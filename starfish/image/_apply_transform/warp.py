from copy import deepcopy
from typing import List, Mapping, Tuple, Union

import numpy as np
import xarray as xr
from skimage import transform
from skimage.transform._geometric import GeometricTransform
from tqdm import tqdm

from starfish.config import StarfishConfig
from starfish.image._apply_transform._base import ApplyTransformBase
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes
from starfish.util import click


class Warp(ApplyTransformBase):

    def __init__(self,
                 transforms_list: Union[str, List[Tuple[Mapping[Axes, int], GeometricTransform]]]
                 ):
        if isinstance(transforms_list, list):
            self.transforms_list = transforms_list
        else:
            # TODO CREATE WAY TO READ TRANSFORMS LIST FROM DISK
            self.transforms_list = list()

    def run(
            self, stack: ImageStack,
            transforms_list: List[Tuple[Mapping[Axes, int], GeometricTransform]],
            in_place: bool=False, verbose: bool=False, **kwargs) -> ImageStack:
        """Applies a transformation to an Imagestack
        Parameters
        ----------
        stack : ImageStack
            Stack to be transformed.
        transforms_list:
            List[Tuple[Mapping[Axes, int], GeometricTransform]] where each entry describes
            a specific axis and the transformation object to apply to it.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on filtering progress (default = False)
        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.
        """
        if not in_place:
            # create a copy of the ImageStack, call apply on that stack with in_place=True
            image_stack = deepcopy(stack)
            return self.run(
                image_stack,
                transforms_list,
                in_place=True,
                **kwargs
            )
        if verbose and StarfishConfig().verbose:
            transforms_list = tqdm(transforms_list)
        all_axes = {Axes.ROUND, Axes.CH, Axes.ZPLANE}
        # TODO CHANGE TO SELF>TRANSFORMS_LIST
        for selector, transformation_object in transforms_list:
            other_axes = all_axes - {list(selector.keys())[0]}
            # iterate through remaining axes
            for axes in stack._iter_axes(other_axes):
                # combine all axes data to select one tile
                selector.update(axes)  # type: ignore
                selected_image, _ = stack.get_slice(selector)
                warped_image = warp(selected_image, transformation_object, **kwargs
                                    ).astype(np.float32)
                stack.set_slice(selector, warped_image)
        return stack

    @staticmethod
    @click.command("Warp")
    @click.pass_context
    def _cli(ctx):
        ctx.obj["component"]._cli_run(ctx, Warp())


def warp(image: Union[xr.DataArray, np.ndarray],
         transformation_object: GeometricTransform,
         **kwargs
         ) -> np.ndarray:
    """ Wrapper around skimage.transform.warp. Warps an image according to a
    given coordinate transformation.
    image: the image to be transformed
    transformation_object: skimage.transform.GeometricTransform
    Returns
    -------
    """
    return transform.warp(image, transformation_object, **kwargs)
