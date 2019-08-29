from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, SpotFindingResults


def build_spot_traces_exact_match(spot_results: SpotFindingResults):
    """
    Combines spots found in matching x/y positions across rounds and channels of
    an ImageStack into traces represented as an IntensityTable.

    Parameters
    -----------
    spot_results: SpotFindingResults
        Spots found accrss rounds/channels of an ImageStack
    """
    # create IntensityTable with same x/y/z info accross all r/ch
    spot_attributes = spot_results[{Axes.ROUND: 0, Axes.CH: 0}]
    ch_labels = spot_results.ch_labels
    round_labels = spot_results.round_labels
    intensity_table = IntensityTable.zeros(
        spot_attributes=spot_attributes,
        ch_labels=ch_labels,
        round_labels=round_labels,
    )
    for r, c in spot_results.keys():
        intensity_table.loc[dict(c=c, r=r)] = \
            spot_results[{Axes.ROUND: r, Axes.CH: c}].data[Features.INTENSITY]
    return intensity_table
