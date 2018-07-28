from typing import Tuple

from starfish.codebook import Codebook
from starfish.image import ImageStack
from starfish.intensity_table import IntensityTable
from starfish.pipeline.features.pixels.combine_adjacent_features import \
    ConnectedComponentDecodingResult, combine_adjacent_features
from ._base import PixelFinderAlgorithmBase


class PixelSpotDetector(PixelFinderAlgorithmBase):

    def __init__(
            self, codebook: Codebook, distance_threshold: float,
            magnitude_threshold: int, min_area: int, max_area: int, norm_order: int=2,
            crop_x: int=0, crop_y: int=0, crop_z: int=0, **kwargs) -> None:
        """Decode an image by first coding each pixel, then combining the results into spots

        Parameters
        ----------
        codebook : Codebook
            Codebook object mapping codewords to the targets they are designed to detect
        distance_threshold : float
            spots whose codewords are more than this distance from an expected code are filtered
        magnitude_threshold : int
            spots with intensity less than this value are filtered
        min_area : int
            spots with total area less than this value are filtered
        max_area : int
            spots with total area greater than this value are filtered
        norm_order : int
            order of L_p norm to apply to intensities and codes when using metric_decode to pair
            each intensities to its closest target (default = 2)
        crop_x, crop_y, crop_z : int
            number of pixels to crop from the top and bottom of each of the x, y, and z axes of
            an ImageStack (default = 0)

        """
        self.codebook = codebook
        self.distance_threshold = distance_threshold
        self.magnitude_threshold = magnitude_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.norm_order = norm_order
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_z = crop_z

    def find(self, stack: ImageStack) \
            -> Tuple[IntensityTable, ConnectedComponentDecodingResult]:
        """decode pixels and combine them into spots using connected component labeling

        Parameters
        ----------
        stack : ImageStack
            ImageStack containing spots

        Returns
        -------
        IntensityTable :
            IntensityTable containing decoded spots
        ConnectedComponentDecodingResult :
            Results of connected component labeling

        """
        pixel_intensities = IntensityTable.from_image_stack(
            stack, crop_x=self.crop_x, crop_y=self.crop_y, crop_z=self.crop_z)
        decoded_intensities = self.codebook.metric_decode(
            pixel_intensities,
            max_distance=self.distance_threshold,
            min_intensity=self.magnitude_threshold,
            norm_order=self.norm_order
        )
        decoded_spots, image_decoding_results = combine_adjacent_features(
            decoded_intensities,
            min_area=self.min_area,
            max_area=self.max_area
        )

        return decoded_spots, image_decoding_results

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--codebook-input", help="csv file containing a codebook")
        group_parser.add_argument(
            "--distance-threshold", default=0.5176,
            help="maximum distance a pixel may be from a codeword before it is filtered")
        group_parser.add_argument("--magnitude-threshold", type=float, default=1, help="minimum magnitude of a feature")
        group_parser.add_argument("--area-threshold", type=float, default=2, help="minimum area of a feature")
        group_parser.add_argument('--crop-x', type=int, default=0)
        group_parser.add_argument('--crop-y', type=int, default=0)
        group_parser.add_argument('--crop-z', type=int, default=0)
