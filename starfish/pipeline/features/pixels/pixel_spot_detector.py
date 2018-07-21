from typing import List, Tuple

import numpy as np


from starfish.codebook import Codebook
from starfish.image import ImageStack
from starfish.intensity_table import IntensityTable
from starfish.pipeline.features.pixels.combine_adjacent_features import \
    ConnectedComponentDecodingResult
from ._base import PixelFinderAlgorithmBase


class PixelSpotDetector(PixelFinderAlgorithmBase):

    def __init__(
            self, codebook: Codebook, distance_threshold: float=0.5176,
            magnitude_threshold: int=1, area_threshold: int=2,
            crop_x: int=0, crop_y: int=0, crop_z: int=0, **kwargs) -> None:
        """Decode an image by first coding each pixel, then combining the results into spots

        Parameters
        ----------
        codebook : Codebook
            Codebook object mapping codewords to the genes they represent
        distance_threshold : float
            spots whose codewords are more than this distance from an expected code are filtered
            (default = 0.5176)
        magnitude_threshold : int
            spots with intensity less than this value are filtered (default 1)
        area_threshold : int
            spots with total area less than this value are filtered
        crop_x, crop_y, crop_z : int
            number of pixels to crop from the top and bottom of each of the x, y, and z axes of
            an ImageStack (default = 0)

        """
        self.codebook = codebook
        self.distance_threshold = distance_threshold
        self.magnitude_threshold = magnitude_threshold
        self.area_threshold = area_threshold
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
        decoded_intensities = self.codebook.decode_euclidean(
            pixel_intensities,
            max_distance=self.distance_threshold,
            min_intensity=self.magnitude_threshold
        )
        decoded_spots, image_decoding_results = decoded_intensities.combine_adjacent_features(
            area_threshold=self.area_threshold,
            assume_contiguous=True
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
