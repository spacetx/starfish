from typing import Mapping, NamedTuple

import numpy as np
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from starfish.codebook.codebook import Codebook
from starfish.types import Axes, Features

class LabelResult(NamedTuple):
    metric_result: np.ndarray
    gene_indices: np.ndarray
    index_to_gene_map: Mapping[int, str]


class Label:

    # TODO add docstring
    def __init__(self, codebook: Codebook, metric: str="euclidean", k_targets: int=1) -> None:
        """

        Parameters
        ----------
        codebook : Codebook
            codebook that will be used to decode image
        metric : str
            the metric used to calculate pixel intensity distance from codes in codebook
        k_targets : int
            number of labeled arrays to emit for each pixel

        """
        self.codebook = codebook
        self.metric = metric
        self.k_targets = k_targets

    @staticmethod
    def _decode(
        image: xr.DataArray,
        codebook: Codebook,
        k_targets: int,
        metric: str="euclidean",
        norm_order: int=2
    ) -> LabelResult:
        """
        """
        # TODO make this respect physical coordinates
        traces = image.xarray.stack(
            features=(Axes.ZPLANE.value, Axes.Y.value, Axes.X.value)
        )
        traces = traces.transpose(Features.AXIS, Axes.CH.value, Axes.ROUND.value)

        # normalize codes and traces
        norm_intensities, norms = codebook._normalize_features(traces, norm_order=norm_order)
        norm_codes, _ = codebook._normalize_features(codebook, norm_order=norm_order)
        expected = norm_codes.stack(codes=(Axes.CH.value, Axes.ROUND.value))
        observed = norm_intensities.stack(codes=(Axes.CH.value, Axes.ROUND.value))

        # calculate distances and labels
        nn = NearestNeighbors(n_neighbors=k_targets, algorithm='ball_tree', metric=metric)
        nn.fit(expected)
        ranked_metric_output, ranked_indices = nn.kneighbors(observed)

        # reshape back into image
        def _create_xarray(data):
            z = observed.features[Axes.ZPLANE.value].values
            y = observed.features[Axes.Y.value].values
            x = observed.features[Axes.X.value].values
            xrdata = xr.DataArray(
                data, dims=("pixels", "ranked_labels"),
                coords={
                    Axes.ZPLANE.value: ("pixels", z),
                    Axes.Y.value: ("pixels", y),
                    Axes.X.value: ("pixels", x)
                }
            )
            xrdata = xrdata.set_index(pixels=(Axes.ZPLANE.value, Axes.Y.value, Axes.X.value))
            unstacked = xrdata.unstack("pixels")
            return unstacked

        ranked_metric_output = _create_xarray(ranked_metric_output)
        ranked_indices = _create_xarray(ranked_indices)

        index_to_gene_map = dict(zip(
            norm_codes.indexes[Features.TARGET].values,
            np.arange(len(norm_codes.indexes[Features.TARGET].values))
        ))

        return LabelResult(ranked_metric_output, ranked_indices, index_to_gene_map)

    def run(self, stack, in_place=False, verbose=False, n_processes=None) -> LabelResult:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        # TODO support multiprocessing
        return self._decode(stack, self.codebook, self.k_targets, self.metric)
