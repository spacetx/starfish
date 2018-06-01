from functools import partial

import numpy

from ._base import FilterAlgorithmBase


class Clip(FilterAlgorithmBase):

    def __init__(self, p_min, p_max, **kwargs):
        """Image clipping filter

        Parameters
        ----------
        p_min : int
            values below this percentile are set to p_min
        p_max : int
            values above this percentile are set to p_max
        kwargs
        """
        self.p_min: int = p_min
        self.p_max: int = p_max

    @classmethod
    def get_algorithm_name(cls):
        return "clip"

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--p-min", default=0, type=int, help="clip intensities below this percentile")
        group_parser.add_argument("--p-max", default=100, type=int, help="clip intensities above this percentile")

    @staticmethod
    def clip(image, p_min: int, p_max: int) -> numpy.ndarray:
        """Clip values of img below and above percentiles p_min and p_max

        Parameters
        ----------
        p_min, int (default 0)
          values below this percentile are set to the value of this percentile
        p_max, int (default 100)
          values above this percentile are set to the value of this percentile

        Notes
        -----
        - Wrapper for np.clip
        - No shifting or transformation to adjust dynamic range is done after
          clipping

        Returns
        -------
        np.ndarray
          Numpy array of same shape as img

        """
        v_min, v_max = numpy.percentile(image, [p_min, p_max])

        # asking for a float percentile clipping value from an integer image will
        # convert to float, so store the dtype so it can be restored
        dtype = image.dtype
        image = image.clip(min=v_min, max=v_max)
        return image.astype(dtype)

    def filter(self, stack) -> None:
        """Perform in-place filtering of an image stack and all contained aux images.

        Parameters
        ----------
        stack : starfish.Stack
            Stack to be filtered.

        """
        clip = partial(self.clip, p_min=self.p_min, p_max=self.p_max)
        stack.image.apply(clip)

        # apply to aux dict too:
        for auxiliary_image in stack.auxiliary_images.values():
            auxiliary_image.apply(clip)
