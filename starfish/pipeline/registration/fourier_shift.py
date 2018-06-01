from starfish.constants import Indices
from ._base import RegistrationAlgorithmBase


class FourierShiftRegistration(RegistrationAlgorithmBase):
    """
    Implements fourier shift registration.  TODO: (dganguli) FILL IN DETAILS HERE PLS.

    Performs a simple translation registration.

    See Also
    --------
    https://en.wikipedia.org/wiki/Phase_correlation
    """
    def __init__(self, upsampling, **kwargs):
        self.upsampling = upsampling

    @classmethod
    def get_algorithm_name(cls):
        return "fourier_shift"

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--upsampling", default=1, type=int, help="Amount of up-sampling")

    def register(self, stack):
        # TODO: (ambrosejcarr) is this the appropriate way of dealing with Z in registration?
        mp = stack.max_proj(Indices.CH, Indices.Z)
        dots = stack.auxiliary_images['dots'].max_proj(Indices.HYB, Indices.CH, Indices.Z)

        for h in range(stack.image.num_hybs):
            # compute shift between maximum projection (across channels) and dots, for each hyb round
            # TODO: make the max projection array ignorant of axes ordering.
            shift, error = compute_shift(mp[h, :, :], dots, self.upsampling)
            print("For hyb: {}, Shift: {}, Error: {}".format(h, shift, error))

            for c in range(stack.image.num_chs):
                for z in range(stack.image.num_zlayers):
                    # apply shift to all zlayers, channels, and hyb rounds
                    indices = {Indices.HYB: h, Indices.CH: c, Indices.Z: z}
                    data, axes = stack.image.get_slice(indices=indices)
                    assert len(axes) == 0
                    result = shift_im(data, shift)
                    stack.image.set_slice(indices=indices, data=result)

        return stack


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
