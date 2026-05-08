import numpy as np
import itertools
from sklearn.neighbors import NearestNeighbors
from starfish.core.codebook.codebook import Codebook
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_to_intensity_table
from starfish.core.spots.DecodeSpots.trace_builders import TRACE_BUILDERS
from starfish.core.types import Number, SpotFindingResults, TraceBuildingStrategies, Features, Axes
from ._base import DecodeSpotsAlgorithm


class QualityBarcodeDecoder(DecodeSpotsAlgorithm):
    """
    Normalizes the magnitudes of the spot intensities (after applying a threshold) and all possible
    channel activation patterns, then assign each spot to the closest binary activation pattern,
    measured by the provided metric. Next select all targets from the codebook with consistent 
    activation patterns. Finally, check, whether the combination of these compatible targets explains
    all of the observed intensities.
    If there is no signal in any channel for at least one round "missing" is assigned.
    If there is additional signal, not explained by the valid barcodes, "(invalid)" is appended.

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
    metric : str
        the metric to use to measure distance. Can be any metric that satisfies the triangle
        inequality that is implemented by :py:mod:`scipy.spatial.distance` (default "euclidean")
    norm_order : int
        the norm to use to normalize the magnitudes of spots and codes (default 2, L2 norm)
    trace_building_strategy: TraceBuildingStrategies
        Defines the strategy for building spot traces to decode across rounds and chs of spot
        finding results.
    anchor_round : Optional[int]
        Only applicable if trace_building_strategy is TraceBuildingStrategies.NEAREST_NEIGHBORS.
        The imaging round against which other rounds will be checked for spots in the same
        approximate pixel location.
    search_radius : Optional[int]
        Only applicable if trace_building_strategy is TraceBuildingStrategies.NEAREST_NEIGHBORS.
        Number of pixels over which to search for spots in other rounds and channels.
    return_original_intensities: bool
        If True returns original intensity values in the DecodedIntensityTable instead of
        normalized ones (default=False)
    min_target_quality: float
        Minimum quality for a target to be included. The quality of a target is defined as the 
        sum of the spot intensities in the target's active channels divided by the number of 
        active channels in the target.
    """

    def __init__(
        self,
        codebook: Codebook,
        max_distance: Number,
        min_intensity: Number,
        norm_order: int = 2,
        metric: str = "euclidean",
        trace_building_strategy: TraceBuildingStrategies = TraceBuildingStrategies.EXACT_MATCH,
        anchor_round: int = 1,
        search_radius: int = 3,
        return_original_intensities: bool = True,
        raw_intensity_threshold: float = 0.05,
        min_target_quality: float = 0.03,
    ) -> None:
        self.codebook = codebook
        self.max_distance = max_distance
        self.min_intensity = min_intensity
        self.norm_order = norm_order
        self.metric = metric
        self.trace_builder = TRACE_BUILDERS[trace_building_strategy]
        self.anchor_round = anchor_round
        self.search_radius = search_radius
        self.return_original_intensities = return_original_intensities
        self.raw_intensity_threshold = raw_intensity_threshold
        self.min_target_quality = min_target_quality

    def run(
        self,
        spots: SpotFindingResults,
        *args
    ) -> DecodedIntensityTable:
        """Decode spots using the MultiBarcode strategy.

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
        
        self.codebook._validate_decode_intensity_input_matches_codebook_shape(intensities)

        # add empty metadata fields and return
        if intensities.sizes[Features.AXIS] == 0:
            return DecodedIntensityTable.from_intensity_table(
                intensities,
                targets=(Features.AXIS, np.empty(0, dtype='U')),
                distances=(Features.AXIS, np.empty(0, dtype=np.float64)),
                passes_threshold=(Features.AXIS, np.empty(0, dtype=bool)))

        st = intensities.copy()
        
        # apply threshold
        st.values[st.values < self.raw_intensity_threshold] = 0

        # normalize
        st.values = np.nan_to_num(st/st.sum(axis=2))

        channels = intensities.sizes[Axes.CH.value]
        rounds = intensities.sizes[Axes.ROUND.value]
        missing_rounds = st.sum([Axes.ROUND.value,Axes.CH.value]).round() < rounds

        # generate all possible binary activation patterns
        count = 0
        codebook_comb = np.empty((2 ** len(self.codebook), rounds, channels), dtype=float)
        for n in range(len(self.codebook) +1):
            for subset in itertools.combinations(self.codebook, n):
                if len(subset) == 0:
                    combined = np.zeros_like(self.codebook[0])
                else:
                    combined = np.clip(np.sum(subset, axis=0), 0, 1)
                codebook_comb[count] = combined
                count += 1
        codebook_comb = np.nan_to_num(codebook_comb[1:]/codebook_comb[1:].sum(axis=2, keepdims=True))
        codebook_comb = [codebook_comb[i].flatten(order = 'C') for i in range(len(codebook_comb))]

        # compute closest binary activation pattern for each spot
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean').fit(codebook_comb)
        for_comp = []
        for n in range(len(st)):
            subset = []
            for vector in st.values[n]:
                for value in vector:
                    subset.append(value)
            for_comp.append(subset)
        metric_outputs, indices = nn.kneighbors(for_comp)
        indices = indices.flatten()
        binarized = np.empty((len(indices), rounds, channels), dtype=bool)
        for i in range(len(indices)):
            binarized[i] = codebook_comb[indices[i]].reshape((-1, rounds, channels)) > 0
        st.values = binarized
        st = st.stack(traces=(Axes.ROUND.value, Axes.CH.value))

        # check compatibility with codebook
        def consistent_with_target(binarized_activation_vectors, target):
            return np.all(binarized_activation_vectors & target == target, axis=binarized_activation_vectors.get_axis_num('traces'))
        
        cbs = self.codebook.stack(traces=(Axes.ROUND.value, Axes.CH.value)) > 0
        consistencies = np.stack([consistent_with_target(st, c).values for c in cbs], axis=1)
        targets = np.array([",".join(cbs.target.values[c]) for c in consistencies])
        
        targets[missing_rounds] = "missing"

        qual_targets = {target: np.empty((len(intensities),), dtype=float) for target in self.codebook.target.values}
        for i in range(len(targets)):
            targets_temp = []
            for t in targets[i].split(","):
                if t in self.codebook.target.values:
                    target_vector = self.codebook.values[self.codebook.target.values == t][0]
                    spot_vector = intensities.values[i]
                    qual_targets[t][i] = (spot_vector * target_vector).sum() / target_vector.sum()
                    if qual_targets[t][i] >= self.min_target_quality:
                        targets_temp.append(t)
            targets[i] = ",".join(targets_temp)

        # only to check whether filter is passed: normalize both the intensities and the codebook
        norm_intensities, norms = self.codebook._normalize_features(intensities, norm_order=self.norm_order)

        # only targets with low distances and high intensities should be retained
        passes_filters = np.logical_and(
            norms >= self.min_intensity,
            metric_outputs.reshape(len(metric_outputs),) <= self.max_distance,
            dtype=bool
        )

        
        norm_metric_outputs = metric_outputs / max(metric_outputs)
        rounded_metric_outputs = [round(i, 3) for i in norm_metric_outputs.reshape(len(norm_metric_outputs),)]
        return_intensities = intensities if self.return_original_intensities else norm_intensities
        return_intensities["binarized"] = ((Features.AXIS, Axes.ROUND.value, Axes.CH.value), binarized)
        # norm_intensities is a DataArray, make it back into an IntensityTable
        output = DecodedIntensityTable.from_intensity_table(
            return_intensities,
            targets=(Features.AXIS, targets),
            distances=(Features.AXIS, rounded_metric_outputs),
            passes_threshold=(Features.AXIS, passes_filters))
        for target in qual_targets:
            output[target+"_quality"] = (Features.AXIS, qual_targets[target])
        return output