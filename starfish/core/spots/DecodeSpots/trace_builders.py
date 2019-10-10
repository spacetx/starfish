from typing import Callable, Mapping

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, SpotFindingResults, TraceBuildingStrategies
from .util import _build_intensity_table, _match_spots, _merge_spots_by_round


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


def build_traces_nearest_neighbors(spot_results: SpotFindingResults, anchor_round: int=1,
                                   search_radius: int=3):
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
    intensity_table = _build_intensity_table(
        per_round_spot_results, distances, indices,
        rounds=spot_results.round_labels,
        channels=spot_results.ch_labels,
        search_radius=search_radius,
        anchor_round=anchor_round
    )
    return intensity_table


TRACE_BUILDERS: Mapping[TraceBuildingStrategies, Callable] = {
    TraceBuildingStrategies.EXACT_MATCH: build_spot_traces_exact_match,
    TraceBuildingStrategies.NEAREST_NEIGHBOR: build_traces_nearest_neighbors
}
