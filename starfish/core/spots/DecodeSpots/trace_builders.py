from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, SpotFindingResults


def build_spot_traces_exact_match(spot_results: SpotFindingResults) -> IntensityTable:
    """
    Combines spots found in matching x/y positions across rounds and channels of
    an ImageStack into traces represented as an IntensityTable.

    Parameters
    -----------
    spot_results: SpotFindingResults
        Spots found accrss rounds/channels of an ImageStack
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
