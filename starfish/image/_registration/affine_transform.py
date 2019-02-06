from copy import deepcopy
from typing import Optional, Tuple, Union

from itertools import product
import numpy as np
from skimage.transform import AffineTransform as AT
from skimage.transform import warp

from starfish.image._filter.util import preserve_float_range
from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes
from starfish.util import click
from ._base import RegistrationAlgorithmBase


class AffineTransform(RegistrationAlgorithmBase):
    """
    Performs an affine transform on the specified indices of an ImageStack

    """
    def __init__(
        self, scale: Union[list, tuple]=None, rotation: float=None,
        shear: float=None, translation: Union[list, tuple]=None,
        rounds=None, channels=None, zplanes=None
            ) -> None:

        """Performs an affine transform on the specified indices of an ImageStack
        Wraps skimage's AffineTransform
        http://scikit-image.org/docs/dev/api/skimage.transform.html

        Parameters
        ----------
        scale : list, tuple
            images are registered to within 1 / upsample_factor of a pixel
        rotation : float
            Rotation angle in radians, counter-clockwise direction is positive
        shear : float
            Shear angle in radians, counter-clockwise direction is positive
        translation : list, tuple
            Number of pixels to translate the image (t_x, t_y)
        rounds : np.ndarray
            list of rounds to apply the transformation to. If omitted, transform
            will be applied to all rounds.
        channels : np.ndarray
            list of channels to apply the transformation to. If omitted, transform
            will be applied to all channels.
        zplanes : np.ndarray
            list of zplanes to apply the transformation to. If omitted, transform
            will be applied to all zplanes.
        """
        self.scale = scale
        self.rotation = rotation
        self.shear = shear
        self.translation = translation
        self.rounds = rounds
        self.channels = channels
        self.zplanes = zplanes

    def run(self, image: ImageStack, in_place: bool=False) -> Optional[ImageStack]:
        """Register an ImageStack against a reference image.

        Parameters
        ----------
        image : ImageStack
            The stack to be registered
        in_place : bool
            If false, return a new registered stack. Else, register in-place (default False)

        Returns
        -------


        """

        if not in_place:
            image = deepcopy(image)

        # Get a list of all indices for an axes without provided indices
        if self.rounds is None:
            self.rounds = np.arange(image.num_rounds)
        if self.channels is None:
            self.channels = np.arange(image.num_chs)
        if self.zplanes is None:
            self.zplanes = np.arange(image.num_zplanes)

        # Make the list of indices to iterate over
        slice_indices = product(self.rounds, self.channels, self.zplanes)

        # Get the transform
        transformation = AT(scale=self.scale, rotation=self.rotation,
                                    shear=self.shear, translation=self.translation)

        # Iterate through the selected tiles and perform the transform
        for r, c, z in slice_indices:
            selector = {Axes.ROUND: r, Axes.CH: c, Axes.ZPLANE: z}

            tile = image.get_slice(selector)[0]
            transformed = warp(tile, transformation)
            image.set_slice(selector, transformed.astype(np.float32))

        if not in_place:
            return image
        return None

    @staticmethod
    @click.command("AffineTransform")
    @click.option("--scale", default=None, type=list, help="Scale factor, (Sx, Sy)")
    @click.option("--rotation", default=None, type=float,
                  help="Rotation angle in radians, counter-clockwise direction is positive.")
    @click.option("--shear", default=None, type=float,
                  help="Shear angle in radians, counter-clockwise direction is positive.")
    @click.option("--translation", default=None, type=list,
                  help="Number of pixels to translate the image (t_x, t_y).")
    @click.option("--rounds", default=None, type=list,
                  help="list of rounds to apply the transformation to.")
    @click.option("--channels", default=None, type=list,
                  help="list of channels to apply the transformation to.")
    @click.option("--zplanes", default=None, type=list,
                  help="list of zplanes to apply the transformation to..")
    @click.pass_context
    def _cli(ctx, scale, rotation, shear, translation, rounds, channels, zplanes):
        ctx.obj["component"]._cli_run(
            ctx, AffineTransform(scale, rotation, shear, translation, rounds, channels, zplanes)
                )
