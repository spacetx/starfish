from typing import Optional, Tuple


from starfish.core.codebook.codebook import Codebook
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_to_intensity_table
from ._base import DetectPixelsAlgorithm
from .combine_adjacent_features import CombineAdjacentFeatures, ConnectedComponentDecodingResult


class PixelSpotDecoder(DetectPixelsAlgorithm):
    """Decode an image by first coding each pixel, then combining the results into spots

    Parameters
    ----------
    codebook : Codebook
        Codebook object mapping codewords to the targets they are designed to detect
    metric : str
        the sklearn metric string to pass to NearestNeighbors
    distance_threshold : float
        spots whose codewords are more than this metric distance from an expected code are
        filtered
    magnitude_threshold : int
        spots with intensity less than this value are filtered
    min_area : int
        spots with total area less than this value are filtered
    max_area : int
        spots with total area greater than this value are filtered
    norm_order : int
        order of L_p norm to apply to intensities and codes when using metric_decode to pair
        each intensities to its closest target (default = 2)
    """
    def __init__(
            self, codebook: Codebook, metric: str, distance_threshold: float,
            magnitude_threshold: int, min_area: int, max_area: int, norm_order: int = 2
    ) -> None:

        self.codebook = codebook
        self.metric = metric
        self.distance_threshold = distance_threshold
        self.magnitude_threshold = magnitude_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.norm_order = norm_order

    def run(
            self,
            primary_image: ImageStack,
            n_processes: Optional[int] = None,
            *args,
    ) -> Tuple[DecodedIntensityTable, ConnectedComponentDecodingResult]:
        """decode pixels and combine them into spots using connected component labeling

        Parameters
        ----------
        primary_image : ImageStack
            ImageStack containing spots
        n_processes : Optional[int]
            The number of processes to use for CombineAdjacentFeatures.
             If None, uses the output of os.cpu_count() (default = None).

        Returns
        -------
        DecodedIntensityTable :
            IntensityTable containing decoded spots
        ConnectedComponentDecodingResult :
            Results of connected component labeling

        """
        pixel_intensities = IntensityTable.from_image_stack(primary_image)
        decoded_intensities = self.codebook.decode_metric(
            pixel_intensities,
            max_distance=self.distance_threshold,
            min_intensity=self.magnitude_threshold,
            norm_order=self.norm_order,
            metric=self.metric
        )
        caf = CombineAdjacentFeatures(
            min_area=self.min_area,
            max_area=self.max_area,
            mask_filtered_features=True
        )
        decoded_spots, image_decoding_results = caf.run(intensities=decoded_intensities,
                                                        n_processes=n_processes)

        transfer_physical_coords_to_intensity_table(image_stack=primary_image,
                                                    intensity_table=decoded_spots)
        return decoded_spots, image_decoding_results
