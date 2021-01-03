from collections import defaultdict
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features, SpotFindingResults


def _match_spots(
    round_dataframes: Dict[int, pd.DataFrame], anchor_round: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ For each spot in anchor round, find the closest spot within search_radius in all rounds.

    Parameters
    ----------
    round_dataframes : Dict[int, pd.DataFrame]
        Output from _merge_spots_by_round, contains mapping of image volumes from each round to
        all the spots detected in them.
    anchor_round : int
        The imaging round to seed the local search from.

    Returns
    -------
    pd.DataFrame
        Spots x rounds dataframe containing the distances to the nearest spot. np.nan if
        no spot is detected within search radius
    """
    reference_df = round_dataframes[anchor_round]
    reference_coordinates = reference_df[[Axes.ZPLANE, Axes.Y, Axes.X]]

    dist = pd.DataFrame(
        data=np.zeros((reference_df.shape[0], len(round_dataframes)), dtype=float),
        columns=list(round_dataframes.keys())
    )
    ind = pd.DataFrame(
        data=np.zeros((reference_df.shape[0], len(round_dataframes)), dtype=np.int32),
        columns=list(round_dataframes.keys())
    )

    # fill data for anchor round; every spot is a perfect match to itself.
    ind[anchor_round] = np.arange(reference_df.shape[0], dtype=np.int32)

    # get spots matching across rounds
    for r in sorted(set(round_dataframes.keys()) - {anchor_round, }):
        query_df = round_dataframes[r]
        query_coordinates = query_df[[Axes.ZPLANE, Axes.Y, Axes.X]]
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(query_coordinates)
        distances, indices = nn.kneighbors(reference_coordinates)
        dist[r] = distances
        ind[r] = indices

    return dist, ind


def _build_intensity_table(
    round_dataframes: Dict[int, pd.DataFrame],
    dist: pd.DataFrame,
    ind: pd.DataFrame,
    channels: Sequence[int],
    rounds: Sequence[int],
    search_radius: int,
    anchor_round: int,
) -> IntensityTable:
    """Construct an intensity table from the results of a local search over detected spots

    Parameters
    ----------
    round_dataframes : Dict[int, pd.DataFrame]
        Output from _merge_spots_by_round, contains mapping of image volumes from each round to
        all the spots detected in them.
    dist, ind : pd.DataFrame
        Output from _match_spots, contains distances and indices to the nearest spot for each
        spot in anchor_round.
    channels, rounds : Sequence[int]
        Channels and rounds present in the ImageStack from which spots were detected.
    search_radius : int
        The maximum (euclidean) distance in pixels for a spot to be considered matching in
        a round subsequent to the anchor round.
    anchor_round : int
        The imaging round to seed the local search from.

    """

    anchor_df = round_dataframes[anchor_round]

    # create empty IntensityTable filled with np.nan
    data = np.full((dist.shape[0], len(channels), len(rounds)), fill_value=np.nan)
    dims = (Features.AXIS, Axes.CH.value, Axes.ROUND.value)
    coords: Mapping[Hashable, Tuple[str, Any]] = {
        Features.SPOT_RADIUS: (Features.AXIS, anchor_df[Features.SPOT_RADIUS]),
        Axes.ZPLANE.value: (Features.AXIS, anchor_df[Axes.ZPLANE]),
        Axes.Y.value: (Features.AXIS, anchor_df[Axes.Y]),
        Axes.X.value: (Features.AXIS, anchor_df[Axes.X]),
        Features.SPOT_ID: (Features.AXIS, np.arange(len(anchor_df))),
        Features.AXIS: (Features.AXIS, np.arange(len(anchor_df))),
        Axes.ROUND.value: (Axes.ROUND.value, rounds),
        Axes.CH.value: (Axes.CH.value, channels)
    }
    intensity_table = IntensityTable(data=data, dims=dims, coords=coords)

    # fill IntensityTable
    for r in rounds:
        # get intensity data and indices
        spot_indices = ind[r]
        intensity_data = round_dataframes[r].loc[spot_indices, Features.INTENSITY]
        channel_index = round_dataframes[r].loc[spot_indices, Axes.CH]
        round_index = np.full(ind.shape[0], fill_value=r, dtype=int)
        feature_index = np.arange(ind.shape[0], dtype=int)

        # mask spots that are outside the search radius
        mask = np.asarray(dist[r] < search_radius)  # indices need not match
        feature_index = feature_index[mask]
        channel_index = channel_index[mask]
        round_index = round_index[mask]
        intensity_data = intensity_data[mask]

        # need numpy indexing to set values in vectorized manner
        intensity_table.values[feature_index, channel_index, round_index] = intensity_data

    return intensity_table


def _merge_spots_by_round(
    spot_results: SpotFindingResults
) -> Dict[int, pd.DataFrame]:
    """Merge DataFrames containing spots from different channels into one DataFrame per round.

    Parameters
    ----------
    spot_results : Dict[Tuple[int, int], pd.DataFrame]
        Output of _find_spots. Dictionary mapping (round, channel) volumes to the spots detected
        in them.

    Returns
    -------
    Dict[int, pd.DataFrame]
        Dictionary mapping round volumes to the spots detected in them. Contains an additional
        column labeled by Axes.CH which identifies the channel in which a given spot was
        detected.

    """

    # add channel information to each table and add it to round_data
    round_data: Mapping[int, List] = defaultdict(list)
    for (r, c), df in spot_results.items():
        df = df.spot_attrs.data
        df[Axes.CH.value] = np.full(df.shape[0], c)
        round_data[r].append(df)

    # create one dataframe per round
    round_dataframes = {
        k: pd.concat(v, axis=0).reset_index().drop('index', axis=1)
        for k, v in round_data.items()
    }

    return round_dataframes
