import numpy

from ._base import FilterAlgorithmBase


class Clip(FilterAlgorithmBase):
    @classmethod
    def from_cli_args(cls, args):
        return cls()

    @classmethod
    def get_algorithm_name(cls):
        return "clip"

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--p_min", default=0, type=float, help="clip intensities below this percentile")
        group_parser.add_argument("--p_max", default=100, type=float, help="clip intensities above this percentile")

    @classmethod
    def filter(cls, image, p_min, p_max, *args, **kwargs):
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
