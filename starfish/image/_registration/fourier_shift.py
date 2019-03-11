from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import fourier_shift
from skimage.feature import register_translation

from starfish.imagestack.imagestack import ImageStack
from starfish.types import Axes
from starfish.util import click
from starfish.util.dtype import preserve_float_range
from ._base import RegistrationAlgorithmBase


class FourierShiftRegistration(RegistrationAlgorithmBase):
    """
    Implements fourier shift registration.  TODO: (dganguli) FILL IN DETAILS HERE PLS.

    Performs a simple translation registration.

    """
    def __init__(self, upsampling: int, reference_stack: Union[str, ImageStack], **kwargs) -> None:
        """Implements fourier shift registrations, which performs a simple translation registration

        Parameters
        ----------
        upsampling : int
            images are registered to within 1 / upsample_factor of a pixel
        reference_stack : ImageStack
            the ImageStack against which this object will register images

        See Also
        --------
        https://en.wikipedia.org/wiki/Phase_correlation

        """
        self.upsampling = upsampling

        # TODO ambrosejcarr: remove the ability to load from string in the constructor, move to CLI
        if isinstance(reference_stack, ImageStack):
            self.reference_stack = reference_stack
        else:
            self.reference_stack = ImageStack.from_path_or_url(reference_stack)

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

        # TODO: (ambrosejcarr) is this the appropriate way of dealing with Z in registration?
        mp = image.max_proj(Axes.CH, Axes.ZPLANE)
        mp_numpy = mp._squeezed_numpy(Axes.CH, Axes.ZPLANE)
        reference_image_mp = self.reference_stack.max_proj(Axes.ROUND, Axes.CH, Axes.ZPLANE)
        reference_image_numpy = reference_image_mp._squeezed_numpy(Axes.ROUND,
                                                                   Axes.CH,
                                                                   Axes.ZPLANE)

        for r in image.axis_labels(Axes.ROUND):
            # compute shift between maximum projection (across channels) and dots, for each round
            # TODO: make the max projection array ignorant of axes ordering.
            shift, error = compute_shift(mp_numpy[r, :, :], reference_image_numpy, self.upsampling)
            print(f"For round: {r}, Shift: {shift}, Error: {error}")

            for c in image.axis_labels(Axes.CH):
                for z in image.axis_labels(Axes.ZPLANE):
                    # apply shift to all zplanes, channels, and imaging rounds
                    selector = {Axes.ROUND: r, Axes.CH: c, Axes.ZPLANE: z}
                    data, axes = image.get_slice(selector=selector)
                    assert len(axes) == 0

                    result = shift_im(data, shift)
                    result = preserve_float_range(result)

                    image.set_slice(selector=selector, data=result)

        if not in_place:
            return image
        return None

    @staticmethod
    @click.command("FourierShiftRegistration")
    @click.option("--upsampling", default=1, type=int, help="Amount of up-sampling")
    @click.option("--reference-stack", required=True, type=click.Path(exists=True),
                  help="The image stack to align the input image stack to.")
    @click.pass_context
    def _cli(ctx, upsampling, reference_stack):
        ctx.obj["component"]._cli_run(ctx, FourierShiftRegistration(upsampling, reference_stack))


def compute_shift(
        im: np.ndarray, ref: np.ndarray, upsample_factor: int=1
) -> Tuple[np.ndarray, float]:
    """calculate subpixel image translation through cross-correlation

    Parameters
    ----------
    im : np.ndarray
        reference image
    ref : np.ndarray
        target image
    upsample_factor : int
        images are registered to within 1 / upsample_factor of a pixel

    Returns
    -------
    np.ndarray :
        shift vector required to register ref
    float :
        translation invariant normalized RMS error
    """
    shift, error, _ = register_translation(im, ref, upsample_factor)
    return shift, error


def shift_im(im: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """register image according to the provided shift values"""
    fim_shift = fourier_shift(np.fft.fftn(im), shift * -1)
    im_shift = np.fft.ifftn(fim_shift)
    return im_shift.real
