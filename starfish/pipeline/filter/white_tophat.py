from ._base import FilterAlgorithmBase


class WhiteTophat(FilterAlgorithmBase):
    """
    Performs "white top hat" filtering of an image to enhance spots. "White top hat filtering" finds spots that are both
    smaller and brighter than their surroundings.

    See Also
    --------
    https://en.wikipedia.org/wiki/Top-hat_transform
    """

    def __init__(self, disk_size, **kwargs):
        """Instance of a white top hat morphological masking filter which masks objects larger than `disk_size`

        Parameters
        ----------
        disk_size : int
            diameter of the morphological masking disk in pixels

        """
        self.disk_size = disk_size

    @classmethod
    def get_algorithm_name(cls):
        return "white_tophat"

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument(
            "--disk-size", default=15, type=int, help="diameter of morphological masking disk in pixels")

    def filter(self, stack) -> None:
        """Perform in-place filtering of an image stack and all contained aux images

        Parameters
        ----------
        stack : starfish.Stack
            Stack to be filtered

        """
        from scipy.ndimage.filters import maximum_filter, minimum_filter
        from skimage.morphology import disk

        def white_tophat(image):
            structuring_element = disk(self.disk_size)
            min_filtered = minimum_filter(image, footprint=structuring_element)
            max_filtered = maximum_filter(min_filtered, footprint=structuring_element)
            filtered_image = image - max_filtered
            return filtered_image

        stack.image.apply(white_tophat)

        # apply to aux dict too.
        for auxiliary_image in stack.auxiliary_images.values():
            auxiliary_image.apply(white_tophat)
