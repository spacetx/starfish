from starfish.core.codebook.codebook import Codebook
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Number
from ._base import DecodeAlgorithmBase


# TODO ambrosejcarr add tests
class MetricDistance(DecodeAlgorithmBase):
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
    ) -> None:
        self.codebook = codebook
        self.max_distance = max_distance
        self.min_intensity = min_intensity
        self.norm_order = norm_order
        self.metric = metric

    def run(
        self,
        intensities: IntensityTable,
        *args
    ) -> IntensityTable:
        """Decode spots by selecting the max-valued channel in each sequencing round

        Parameters
        ----------
        intensities : IntensityTable
            IntensityTable to be decoded
        codebook : Codebook
            Contains codes to decode IntensityTable

        Returns
        -------
        IntensityTable :
            IntensityTable decoded and appended with Features.TARGET and Features.QUALITY values.

        """
        return self.codebook.decode_metric(
            intensities,
            max_distance=self.max_distance,
            min_intensity=self.min_intensity,
            norm_order=self.norm_order,
            metric=self.metric,
        )
