from typing import Union

from starfish.constants import Indices
from starfish.image import ImageStack
from starfish.util.argparse import FsExistsType
from ._base import RegistrationAlgorithmBase


class FourierShiftRegistration(RegistrationAlgorithmBase):
    """
    Implements fourier shift registration.  TODO: (dganguli) FILL IN DETAILS HERE PLS.

    Performs a simple translation registration.

    See Also
    --------
    https://en.wikipedia.org/wiki/Phase_correlation
    """
    def __init__(self, upsampling: int, reference_stack: Union[str, ImageStack], **kwargs) -> None:
        self.upsampling = upsampling
        if isinstance(reference_stack, ImageStack):
            self.reference_stack = reference_stack
        else:
            self.reference_stack = ImageStack.from_path_or_url(reference_stack)

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--upsampling", default=1, type=int, help="Amount of up-sampling")
        group_parser.add_argument(
            "--reference-stack", type=FsExistsType(), required=True,
            help="The image stack to align the input image stack to.")

    def register(self, image: ImageStack):
        # TODO: (ambrosejcarr) is this the appropriate way of dealing with Z in registration?
        mp = image.max_proj(Indices.CH, Indices.Z)
        reference_image = self.reference_stack.max_proj(Indices.ROUND, Indices.CH, Indices.Z)

        for h in range(image.num_rounds):
            # compute shift between maximum projection (across channels) and dots, for each round
            # TODO: make the max projection array ignorant of axes ordering.
            shift, error = compute_shift(mp[h, :, :], reference_image, self.upsampling)
            print("For round: {}, Shift: {}, Error: {}".format(h, shift, error))

            for c in range(image.num_chs):
                for z in range(image.num_zlayers):
                    # apply shift to all zlayers, channels, and imaging rounds
                    indices = {Indices.ROUND: h, Indices.CH: c, Indices.Z: z}
                    data, axes = image.get_slice(indices=indices)
                    assert len(axes) == 0
                    result = shift_im(data, shift)
                    image.set_slice(indices=indices, data=result)

        return image


def compute_shift(im, ref, upsample_factor=1):
    from skimage.feature import register_translation

    shift, error, diffphase = register_translation(im, ref, upsample_factor)
    return shift, error


def shift_im(im, shift):
    import numpy as np
    from scipy.ndimage import fourier_shift

    fim_shift = fourier_shift(np.fft.fftn(im), map(lambda x: -x, shift))
    im_shift = np.fft.ifftn(fim_shift)
    return im_shift.real
