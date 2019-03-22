from copy import deepcopy
from typing import Union

import numpy as np
import xarray as xr
from skimage import transform
from skimage.transform._geometric import GeometricTransform
from tqdm import tqdm

from starfish.config import StarfishConfig
from starfish.image._apply_transform._base import ApplyTransformBase
from starfish.image._learn_transform.transforms_list import TransformsList
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes
from starfish.util import click


class Warp(ApplyTransformBase):
    """Class that applies a list of arbitrary skimage GeometricTransforms to an ImageStack
     using skimage.transform.warp"""

    def __init__(self, transforms_list: TransformsList):
        """
        Parameters
        ----------
        transforms_list: TransformsList
            A list of tuples. Each tuple consists of an ImageStack axis and a GeometricTransform
            to apply to the portion of the image identified by the selector.
        """
        self.transforms_list = transforms_list

    def run(self, stack: ImageStack,
            in_place: bool=False, verbose: bool=False, **kwargs) -> ImageStack:
        """Applies a list of transformations to an ImageStack

        Parameters
        ----------
        stack : ImageStack
            Stack to be transformed.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on filtering progress (default = False)

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of the transforms as a new stack.
            Otherwise return the original stack.
        """
        if not in_place:
            # create a copy of the ImageStack, call apply on that stack with in_place=True
            image_stack = deepcopy(stack)
            return self.run(image_stack, in_place=True, **kwargs)
        if verbose and StarfishConfig().verbose:
            self.transforms_list.transforms = tqdm(self.transforms_list.transforms)
        all_axes = {Axes.ROUND, Axes.CH, Axes.ZPLANE}
        for selector, _, transformation_object in self.transforms_list.transforms:
            other_axes = all_axes - set(selector.keys())
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
    @click.option("--transformation-list", required=True, type=click.Path(exists=True),
                  help="The list of transformations to apply to the ImageStack.")
    @click.pass_context
    def _cli(ctx, transformation_list):
        ctx.obj["component"]._cli_run(ctx, Warp(
            TransformsList.from_json(filename=transformation_list)))


def warp(image: Union[xr.DataArray, np.ndarray],
         transformation_object: GeometricTransform,
         **kwargs
         ) -> np.ndarray:
    """ Wrapper around skimage.transform.warp. Warps an image according to a
    given coordinate transformation.

    Parameters
    ----------
    image: np.ndarray
        The image to be transformed
    transformation_object: skimage.transform.GeometricTransform
        The transformation object to apply.

    Returns
    -------
    np.ndarray:
        the warped image.
    """
    return transform.warp(image, transformation_object, **kwargs)
