from trackpy import bandpass

from ._base import FilterAlgorithmBase


class Bandpass(FilterAlgorithmBase):
    @classmethod
    def from_cli_args(cls, args):
        return cls()

    @classmethod
    def get_algorithm_name(cls):
        return "bandpass"

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--lshort", default=0.5, type=float, help="filter signals below this frequency")
        group_parser.add_argument("--llong", default=7, type=int, help="filter signals above this frequency")
        group_parser.add_argument("--threshold", default=1, type=int, help="clip pixels below this intensity value")
        group_parser.add_argument("--truncate", default=4, type=int)

    @classmethod
    def filter(cls, image, lshort=0.5, llong=7, threshold=1, truncate=4):
        """Apply a bandpass filter to remove noise and background variation

        :param np.ndarray image: Image to filter
        :param float lshort: filter frequencies below this value
        :param int llong: filter frequencies above this odd integer value
        :param float threshold: zero any pixels below this intensity value
        :param float truncate:  # todo document

        """
        bandpassed = bandpass(
            image, lshort=lshort, llong=llong, threshold=threshold,
            truncate=truncate
        )
        return bandpassed
