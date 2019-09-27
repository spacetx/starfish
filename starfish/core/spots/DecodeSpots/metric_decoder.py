from starfish.core.codebook.codebook import Codebook
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_to_intensity_table
from starfish.core.spots.DecodeSpots.trace_builders import trace_builders
from starfish.core.types import Number, SpotFindingResults, TraceBuildingStrategies
from ._base import DecodeSpotsAlgorithm


class MetricDistance(DecodeSpotsAlgorithm):
    """
    Normalizes both the magnitudes of the codes and the spot intensities, then decodes spots by
    assigning each spot to the closest code, measured by the provided metric.

    Codes greater than max_distance from the nearest code, or dimmer than min_intensity, are
    discarded.

    Parameters
    ----------
    codebook : Codebook
        codebook containing targets the experiment was designed to quantify
    max_distance : Number
        spots greater than this distance from their nearest target are not decoded
    min_intensity : Number
        spots dimmer than this intensity are not decoded
    norm_order : int
        the norm to use to normalize the magnitudes of spots and codes (default 2, L2 norm)
    metric : str
        the metric to use to measure distance. Can be any metric that satisfies the triangle
        inequality that is implemented by :py:mod:`scipy.spatial.distance` (default "euclidean")
    """

    def __init__(
        self,
        codebook: Codebook,
        max_distance: Number,
        min_intensity: Number,
        norm_order: int = 2,
        metric: str = "euclidean",
        anchor_round: int = 1,
        search_radius: int = 3,
        trace_building_strategy: TraceBuildingStrategies = TraceBuildingStrategies.EXACT_MATCH
    ) -> None:
        self.codebook = codebook
        self.max_distance = max_distance
        self.min_intensity = min_intensity
        self.norm_order = norm_order
        self.metric = metric
        self.trace_builder = trace_builders[trace_building_strategy]
        self.anchor_round = anchor_round
        self.search_radius = search_radius

    def run(
        self,
        spots: SpotFindingResults,
        *args
    ) -> DecodedIntensityTable:
        """Decode spots by selecting the max-valued channel in each sequencing round

        Parameters
        ----------
        spots : SpotFindingResults
            A Dict of tile indices and their corresponding measured spots

        Returns
        -------
        DecodedIntensityTable :
            IntensityTable decoded and appended with Features.TARGET and Features.QUALITY values.

        """

        intensities = self.trace_builder(spot_results=spots,
                                         anchor_round=self.anchor_round,
                                         search_radius=self.search_radius)
        transfer_physical_coords_to_intensity_table(intensity_table=intensities, spots=spots)
        return self.codebook.decode_metric(
            intensities,
            max_distance=self.max_distance,
            min_intensity=self.min_intensity,
            norm_order=self.norm_order,
            metric=self.metric,
        )
