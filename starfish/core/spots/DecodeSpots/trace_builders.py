from typing import Callable, Mapping

import numpy as np
import pandas as pd

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, SpotAttributes, SpotFindingResults, \
    TraceBuildingStrategies
from .util import (
    _build_intensity_table_graph_results,
    _build_intensity_table_nearest_neighbor_results,
    _build_spot_traces_per_round,
    _compute_spot_trace_qualities,
    _match_spots,
    _merge_spots_by_round
)


def build_spot_traces_exact_match(spot_results: SpotFindingResults, **kwargs) -> IntensityTable:
    """
    Combines spots found in matching x/y positions across rounds and channels of
    an ImageStack into traces represented as an IntensityTable.

    Parameters
    -----------
    spot_results: SpotFindingResults
        Spots found across rounds/channels of an ImageStack
    """
    # create IntensityTable with same x/y/z info accross all r/ch
    spot_attributes = list(spot_results.values())[0]
    intensity_table = IntensityTable.zeros(
        spot_attributes=spot_attributes,
        round_labels=spot_results.round_labels,
        ch_labels=spot_results.ch_labels,
    )
    for r, c in spot_results.keys():
        value = spot_results[{Axes.ROUND: r, Axes.CH: c}].data[Features.INTENSITY]
        # if no exact match set value to 0
        value = 0 if value.empty else value
        intensity_table.loc[dict(c=c, r=r)] = value
    return intensity_table


def build_traces_sequential(spot_results: SpotFindingResults, **kwargs) -> IntensityTable:
    """
    Build spot traces  without merging across channels and imaging rounds. Used for sequential
    methods like smFIsh.

    Parameters
    ----------
    spot_results: SpotFindingResults
        Spots found across rounds/channels of an ImageStack

    Returns
    -------
    IntensityTable :
        concatenated input SpotAttributes, converted to an IntensityTable object

    """

    all_spots = pd.concat([sa.data for sa in spot_results.values()], sort=True)

    intensity_table = IntensityTable.zeros(
        spot_attributes=SpotAttributes(all_spots),
        ch_labels=spot_results.ch_labels,
        round_labels=spot_results.round_labels,
    )

    i = 0
    for (r, c), attrs in spot_results.items():
        for _, row in attrs.data.iterrows():
            selector = dict(features=i, c=c, r=r)
            intensity_table.loc[selector] = row[Features.INTENSITY]
            i += 1
    return intensity_table


def build_traces_nearest_neighbors(
        spot_results: SpotFindingResults,
        anchor_round: int=1,
        search_radius: int=3,
        **kwargs
):
    """
    Combine spots found across round and channels of an ImageStack using a nearest neighbors
    strategy

    Parameters
    -----------
    spot_results : SpotFindingResults
        Spots found across rounds/channels of an ImageStack
    anchor_round : int
        The imaging round against which other rounds will be checked for spots in the same
        approximate pixel location.
    search_radius : int
        Number of pixels over which to search for spots in other rounds and channels.

    """
    per_round_spot_results = _merge_spots_by_round(spot_results)

    distances, indices = _match_spots(
        per_round_spot_results,
        anchor_round=anchor_round
    )
    intensity_table = _build_intensity_table_nearest_neighbor_results(
        per_round_spot_results, distances, indices,
        rounds=spot_results.round_labels,
        channels=spot_results.ch_labels,
        search_radius=search_radius,
        anchor_round=anchor_round
    )
    return intensity_table


def build_traces_graph_based(
        spot_results: SpotFindingResults,
        k_d: float,
        search_radius: int,
        search_radius_max: int,
        anchor_round: int,
        **kwargs
):
    """
    Overlapping spots are merged across channels within each round in order to handle fluorescent
    bleed-trough. Next, a quality score is assigned for each detected spot (maximum intensity
    divided by intensity vector l2-norm). Detected spots belonging to different sequencing rounds
    and closer than d_th are connected in a graph, forming connected components of spot detections.
    Next, for each connected component, edges between not connected spots belonging to consecutive
    rounds are forced if they are closer than dth_max. Finally, all the edges that connect spots
    non belonging to consecutive rounds are removed and each connected component is solved by
    maximum flow minimum cost algorithm. Costs are inversely proportional to spot quality and
    distances. The final intensity table is then initialized with the intensity table of the
    round chosen as anchor (default: first round).


    Parameters
    ----------
    spot_results : SpotFindingResults
        Spots found across rounds/channels of an ImageStack
    anchor_round : int
        The imaging round against which other rounds will be checked for spots in the same
        approximate pixel location.
    search_radius : int
        Euclidean distance in pixels over which to search for spots in subsequent rounds.
    search_radius_max : int
        The maximum (euclidian) distance in pixels allowed between nodes belonging to the
        same sequence
    k_d : float
        distance weight

    Notes
    -----
    [1] Partel, G. et al. Identification of spatial compartments in tissue from in situ sequencing
    data. BioRxiv, https://doi.org/10.1101/765842, (2019).
    """
    if spot_results.count_total_spots() == 0:
        spot_attributes = list(spot_results.values())[0]
        return IntensityTable.zeros(
            spot_attributes=spot_attributes,
            round_labels=spot_results.round_labels,
            ch_labels=spot_results.ch_labels,
        )
    else:
        round_dataframes = _merge_spots_by_round(spot_results)

        spot_traces = _build_spot_traces_per_round(
            round_dataframes,
            channels=spot_results.ch_labels,
            rounds=spot_results.round_labels)

        spot_traces = _compute_spot_trace_qualities(spot_traces)

        intensity_table = _build_intensity_table_graph_results(
            intensity_tables=spot_traces,
            rounds=spot_results.round_labels,
            search_radius=search_radius,
            search_radius_max=search_radius_max,
            k_d=k_d,
            anchor_round=anchor_round)

        # Drop intensities with empty rounds
        drop = [np.any(np.all(np.isnan(intensity_table.values[x, :, :]), axis=0))
                for x in range(intensity_table.shape[0])]
        intensity_table = IntensityTable(
            intensity_table[np.arange(intensity_table.shape[0])[np.invert(drop)]])

        return intensity_table


TRACE_BUILDERS: Mapping[TraceBuildingStrategies, Callable] = {
    TraceBuildingStrategies.EXACT_MATCH: build_spot_traces_exact_match,
    TraceBuildingStrategies.NEAREST_NEIGHBOR: build_traces_nearest_neighbors,
    TraceBuildingStrategies.SEQUENTIAL: build_traces_sequential,
    TraceBuildingStrategies.GRAPH: build_traces_graph_based
}
