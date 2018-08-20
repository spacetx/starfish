from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import fourier_shift
from skimage.feature import register_translation

from starfish.stack import ImageStack
from starfish.types import Indices
from starfish.util.argparse import FsExistsType
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

    @classmethod
    def add_arguments(cls, group_parser) -> None:
        group_parser.add_argument("--upsampling", default=1, type=int, help="Amount of up-sampling")
        group_parser.add_argument(
            "--reference-stack", type=FsExistsType(), required=True,
            help="The image stack to align the input image stack to.")

    def run(self, image: ImageStack, in_place: bool=True) -> Optional[ImageStack]:
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
        mp = image.max_proj(Indices.CH, Indices.Z)
        reference_image = self.reference_stack.max_proj(Indices.ROUND, Indices.CH, Indices.Z)

        for r in range(image.num_rounds):
            # compute shift between maximum projection (across channels) and dots, for each round
            # TODO: make the max projection array ignorant of axes ordering.
            shift, error = compute_shift(mp[r, :, :], reference_image, self.upsampling)
            print(f"For round: {r}, Shift: {shift}, Error: {error}")

            for c in range(image.num_chs):
                for z in range(image.num_zlayers):
                    # apply shift to all zlayers, channels, and imaging rounds
                    indices = {Indices.ROUND: r, Indices.CH: c, Indices.Z: z}
                    data, axes = image.get_slice(indices=indices)
                    assert len(axes) == 0
                    result = shift_im(data, shift)
                    image.set_slice(indices=indices, data=result)

        if not in_place:
            return image
        return None


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
