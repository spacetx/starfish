from itertools import product


from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, SpotFindingResults


def build_spot_traces_exact_match(spot_results: SpotFindingResults):
    # create IntensityTable with same x/y/z info accross all r/ch
    spot_attributes = spot_results.get_spots_for_round_ch({Axes.ROUND: 0, Axes.CH: 0})
    ch_labels = spot_results.round_labels
    round_labels = spot_results.ch_labels
    intensity_table = IntensityTable.zeros(
        spot_attributes=spot_attributes,
        ch_labels=ch_labels,
        round_labels=round_labels,
    )
    indices = product(ch_labels, round_labels)
    for c, r in indices:
        intensity_table.loc[dict(c=c, r=r)] = \
            spot_results.get_spots_for_round_ch({Axes.ROUND: r, Axes.CH: c}).data[Features.INTENSITY]
    return intensity_table



