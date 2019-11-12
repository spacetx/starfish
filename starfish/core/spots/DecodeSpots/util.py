from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

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


def _build_intensity_table_nearest_neighbor_results(
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


def _build_spot_traces_per_round(
        round_dataframes: Dict[int, pd.DataFrame],
        channels: Sequence[int],
        rounds: Sequence[int]
) -> Dict[int, IntensityTable]:
    """ For each round, find connected components of spots across channels and merge them
    in a single spot trace.

    Parameters
    ----------
    round_dataframes : Dict[int, pd.DataFrame]
        Output from _merge_spots_by_round, contains mapping of image volumes from each round to
        all the spots detected in them.
    channels, rounds : Sequence[int]
        Channels and rounds present in the ImageStack from which spots were detected.
    Returns
    -------
    Dict[int, IntensityTable]
        Dictionary mapping round to the relative IntensityTable.
    """
    intensity_tables = {}

    # get spots matching across channels
    for r, df in round_dataframes.items():
        # Find connected components across channels
        G = nx.Graph()
        kdT = KDTree(df.loc[:, [Axes.ZPLANE.value, Axes.Y.value, Axes.X.value]].values)
        pairs = kdT.query_pairs(1, p=1)
        G.add_nodes_from(df.index.values)
        G.add_edges_from(pairs)
        conn_comps = [list(i) for i in nx.connected_components(G)]
        # for each connected component keep detection with highest intensity
        refined_conn_comps = []
        for i in range(len(conn_comps)):
            df_tmp = df.loc[conn_comps[i], :]
            kdT_tmp = KDTree(df_tmp.loc[:, [Axes.ZPLANE.value, Axes.Y.value, Axes.X.value]].values)
            # Check if all spots whitin a conn component are at most 1 pixels away
            # from each other (Manhattan distance)
            spot_pairs = len(list(combinations(np.arange(len(df_tmp)), 2)))
            spots_connected = len(kdT_tmp.query_pairs(2, p=1))  # 2 could be a parameter
            if spot_pairs == spots_connected:
                # Merge spots
                refined_conn_comps.append(conn_comps[i])
            else:
                # split non overlapping signals
                for j, row in df_tmp.drop_duplicates([Axes.ZPLANE.value, Axes.Y.value,
                                                      Axes.X.value]).iterrows():
                    refined_conn_comps.append(df_tmp[(df_tmp.z == row.z) & (df_tmp.y == row.y)
                                                     & (df_tmp.x == row.x)].index.values.tolist())

        data = np.full((len(refined_conn_comps), len(channels), len(rounds)),
                       fill_value=np.nan)
        spot_radius = []
        z = []
        y = []
        x = []
        for f_idx, s in enumerate(refined_conn_comps):
            df_tmp = df.loc[s]
            anchor_s_idx = df_tmp.intensity.idxmax()
            z.append(df_tmp.loc[anchor_s_idx, Axes.ZPLANE.value])
            y.append(df_tmp.loc[anchor_s_idx, Axes.Y.value])
            x.append(df_tmp.loc[anchor_s_idx, Axes.X.value])
            spot_radius.append(df_tmp.loc[anchor_s_idx, Features.SPOT_RADIUS])
            for i, row in df_tmp.iterrows():
                data[f_idx, int(row.c), r] = row.intensity
        # # create IntensityTable
        dims = (Features.AXIS, Axes.CH.value, Axes.ROUND.value)
        coords: Mapping[Hashable, Tuple[str, Any]] = {
            Features.SPOT_RADIUS: (Features.AXIS, spot_radius),
            Axes.ZPLANE.value: (Features.AXIS, z),
            Axes.Y.value: (Features.AXIS, y),
            Axes.X.value: (Features.AXIS, x),
            Axes.ROUND.value: (Axes.ROUND.value, rounds),
            Axes.CH.value: (Axes.CH.value, channels)
        }
        intensity_table = IntensityTable(data=data, dims=dims, coords=coords)
        intensity_tables[r] = intensity_table

    return intensity_tables


def _baseCalling(data: list, rounds: Sequence[int], search_radius_max: int) -> np.ndarray:
    """Extract intensity table feature indeces and channels from each connected component graph

    Parameters
    ----------
    data : list
        Output from _runGraphDecoder, contains decoded spots
    rounds : Sequence[int]
        Rounds present in the ImageStack from which spots were detected
    search_radius_max : int
        The maximum (euclidian) distance in pixels allowed between nodes belonging
        to the same sequence

    Returns
    -------
    np.ndarray
        feature indeces arrays of _merge_spots_by_round output intensity tables ordered by round
    """
    idx = []
    if data:
        for graph in tqdm(data):
            G = graph['G']
            Dvar = graph['Dvar']
            for c in nx.connected_components(G):
                c = np.array(list(c))
                c = c[c <= Dvar.X_idx.max()]
                Dvar_c = Dvar[(Dvar.X_idx.isin(c))]
                if len(Dvar_c) == len(rounds):
                    k1 = KDTree(Dvar_c[[Axes.X.value, Axes.Y.value, Axes.ZPLANE.value]].values)
                    max_d = np.amax(list(k1.sparse_distance_matrix(k1, np.inf).values()))
                    if max_d <= search_radius_max:
                        idx.append(Dvar[
                            (Dvar.X_idx.isin(c))].sort_values(['r']).feature_id.values)
    return np.array(idx).astype(np.uint)


def _compute_spot_trace_qualities(intensity_tables: Dict[int, IntensityTable]
                                  ) -> Dict[int, IntensityTable]:
    """Interate over the intesity tables of each round and assign to each feature a quality score
    Parameters
    ----------

    Returns
    -------
    Dict[int,IntensityTable]:
        Dictionary mapping round to the relative IntensityTable with quality coordinate Q
        representing the quality score of each feature.
    """
    for i in intensity_tables:
        intensity_tables[i]['Q'] = (Features.AXIS,
                                    np.divide(np.amax(intensity_tables[i].fillna(0).values,
                                              axis=1),
                                              np.linalg.norm(
                                              intensity_tables[i].fillna(0).values,
                                              2, axis=1),
                                              where=np.linalg.norm(
                                              intensity_tables[i].fillna(0).values,
                                              2, axis=1) != 0)[:, i])
    return intensity_tables


def _runGraphBuilder(data: pd.DataFrame,
                     d_th: float,
                     k_d: float,
                     dth_max: float) -> list:
    """Find connected components of detected spots across rounds and call the graph
    decoder for each connected component instance.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of detected spots with probability values, with columns
        [x, y, z, r, c, idx, p0, p1, feature_id]
    d_th : flaot
         maximum distance inside connected component between two connected spots
    k_d : float
        distance weight
    dth_max : float
        maximum distance inside connected component between every pair of spots

    Returns
    -------
    list[Dict[str,Any]]
        List of dictionaries output of _runMaxFlowMinCost
    """
    print("Generate Graph Model...\n")
    num_hyb = np.arange(0, int(np.amax(data.r)) + 1)
    data.sort_values('r', inplace=True)
    data = data.reset_index(drop=True)
    # Graphical Model Data Structures
    # Generate connected components
    G = nx.Graph()
    G.add_nodes_from(data.index.values)
    for h1 in tqdm(num_hyb):
        KDTree_h1 = KDTree(data[data.r == h1][[Axes.X.value, Axes.Y.value, Axes.ZPLANE.value]])
        for h2 in num_hyb[h1:]:
            if h1 != h2:
                KDTree_h2 = KDTree(data[data.r == h2][[Axes.X.value, Axes.Y.value,
                                                       Axes.ZPLANE.value]])
                query = KDTree_h1.query_ball_tree(KDTree_h2, d_th, p=2)
                E = []
                offset1 = data.index[data.r == h1].min()
                offset2 = data.index[data.r == h2].min()
                for i, e1 in enumerate(query):
                    if e1:
                        for e2 in e1:
                            E.append((i + offset1, e2 + offset2))
                G.add_edges_from(E)

    conn_comps = [list(c) for c in nx.connected_components(G)]
    for c in tqdm(range(len(conn_comps))):
        data.loc[conn_comps[c], 'conn_comp'] = c

    # Drop conn components with less than n_hybs elements
    gr = data.groupby('conn_comp')
    for i, group in gr:
        if len(group) < len(num_hyb):
            data = data.drop(group.index)
    labels = np.unique(data.conn_comp)

    if labels.size > 0:
        print("Run Graph Model...\n")
        res = []
        for l in tqdm(np.nditer(labels), total=len(labels)):
            res.append(_runMaxFlowMinCost(data, int(l), d_th, k_d, num_hyb, dth_max))
        # return maxFlowMinCost
        return [x for x in res if x['G'] is not None]
    else:
        return []


def _prob2Eng(p: float) -> float:
    """Convert probability values into energy by inverse Gibbs distribution

    Parameters
    ----------
    p : float
        probability value

    Returns
    -------
    float
        energy value
    """
    return -1.0 * np.log(np.clip(p, 0.00001, 0.99999))


def _runMaxFlowMinCost(
        data: pd.DataFrame,
        l: int,
        d_th: float,
        k_d: float,
        rounds: np.array,
        dth_max: float) -> Dict:
    """Build the graph model for the given connected component and solve the graph
    with max flow min cost alghorithm

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of detected spots with probability values, with columns
        [x, y, z, r, c, idx, p0, p1]
    l : int
        connected component index
    d_th : float
        maximum distance inside connected component between two connected spots
    k_d : float
        distance weight
    rounds : np.array[int]
        Channels and rounds present in the ImageStack from which spots were detected.
    dth_max : float
        maximum distance inside connected component between every pair of spots

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping the decoded graph, Dataframe of detected spots with
        probability values, Dataframe of connected spots
    """

    if len(data[data.conn_comp == l]):
        if len(np.unique(data[data.conn_comp == l].r)) == len(rounds):
            data_tmp = data[data.conn_comp == l].sort_values(['r']).copy()
            data_tmp.reset_index(inplace=True, drop=True)
            Dvar_tmp = data_tmp.loc[:, [Axes.X.value, Axes.Y.value, Axes.ZPLANE.value,
                                        Axes.ROUND.value, Axes.CH.value, 'feature_id']]
            Dvar_tmp['E_0'] = data_tmp.p0.apply(_prob2Eng)
            Dvar_tmp['E_1'] = data_tmp.p1.apply(_prob2Eng)
            Dvar_tmp['X_idx'] = data_tmp.index.values

            X_idx_tmp = len(Dvar_tmp)
            Tvar_tmp = pd.DataFrame(
                data={'x_idx': [],
                      'anchestor_x_idx': [],
                      'descendant_x_idx': [],
                      'E_0': [],
                      'E_1': [],
                      'mu_d': []})
            for h1 in rounds[:-1]:
                h2 = h1 + 1

                df1 = data_tmp[data_tmp.r == h1]
                df2 = data_tmp[data_tmp.r == h2]
                df1_coords = df1[[Axes.X.value, Axes.Y.value, Axes.ZPLANE.value]].values
                df2_coords = df2[[Axes.X.value, Axes.Y.value, Axes.ZPLANE.value]].values

                KDTree_h1 = KDTree(df1_coords)
                KDTree_h2 = KDTree(df2_coords)
                query = KDTree_h1.query_ball_tree(KDTree_h2, dth_max, p=2)
                for i in range(len(query)):
                    if len(query[i]):
                        X_idx = [(X_idx_tmp + x) for x in range(len(query[i]))]
                        d = [np.linalg.norm(df1_coords[i] - df2_coords[x]) for x in query[i]]
                        mu_d = [1 / (1 + k_d * x) for x in d]

                        Tvar_tmp = Tvar_tmp.append(
                            pd.DataFrame(data={
                                'x_idx': X_idx,
                                'anchestor_x_idx': np.ones(len(query[i])) * df1.index[i],
                                'descendant_x_idx': df2.index[query[i]].values,
                                'E_0': [_prob2Eng(1 - x) for x in mu_d],
                                'E_1': [_prob2Eng(x) for x in mu_d],
                                'mu_d': mu_d}))
                        X_idx_tmp = X_idx[-1] + 1

            Dvar_tmp.X_idx = Dvar_tmp.X_idx + 1
            Tvar_tmp.anchestor_x_idx = Tvar_tmp.anchestor_x_idx + 1
            Tvar_tmp.descendant_x_idx = Tvar_tmp.descendant_x_idx + 1

            Dvar_tmp['X'] = np.arange(1, len(Dvar_tmp) + 1)

            sink = Dvar_tmp.X.max() + 1

            # Inizialize graph
            G = nx.DiGraph()

            E = []  # Edges
            n = sink + 1

            for h in rounds:
                for idx, row in Dvar_tmp[(Dvar_tmp.r == h)].iterrows():
                    if h == 0:
                        E.append((0, row.X, {'capacity': 1, 'weight': 0}))
                    # Add detection edges
                    E.append((row.X, n, {
                        'capacity': 1,
                        'weight': np.round(row.E_1 * 1000000).astype(int)}))
                    n = n + 1

                G.add_edges_from(E)
                E = []
                for idx, row in Tvar_tmp[(
                        Tvar_tmp.anchestor_x_idx.isin(
                            Dvar_tmp[Dvar_tmp.r == h].X_idx))].iterrows():
                    # Add transition edges
                    E.append((list(G.successors(row.anchestor_x_idx))[0], row.descendant_x_idx,
                              {'capacity': 1,
                               'weight': np.round(row.E_1 * 1000000).astype(int)}))
                G.add_edges_from(E)

            # For each D of last cycle connect to sink
            E = []
            for idx, row in Dvar_tmp[(Dvar_tmp.r == rounds.max())].iterrows():
                E.append((list(G.successors(row.X_idx))[0], sink, {'capacity': 1, 'weight': 0}))
            G.add_edges_from(E)

            # Prune graph removing leaf nodes
            remove_nodes = []
            for n in G.nodes:
                n_set = nx.algorithms.descendants(G, n)
                if sink not in n_set:
                    remove_nodes.append(n)
                    if n == 0:  # source and sink are not connected
                        return {'G': None, 'Dvar': None, 'Tvar': None}

            remove_nodes.remove(sink)
            G.remove_nodes_from(remove_nodes)

            MaxFlowMinCost = nx.max_flow_min_cost(G, 0, sink)
            # Decode sequence
            E = []
            for n1 in MaxFlowMinCost:
                for n2 in MaxFlowMinCost[n1]:
                    if MaxFlowMinCost[n1][n2] == 1:
                        E.append((int(n1), n2, {}))
            G = nx.Graph()
            G.add_edges_from(E)
            G.remove_node(0)
            G.remove_node(sink)

            return {'G': G, 'Dvar': Dvar_tmp, 'Tvar': Tvar_tmp}
        else:
            return {'G': None, 'Dvar': None, 'Tvar': None}
    else:
        return {'G': None, 'Dvar': None, 'Tvar': None}


def _build_intensity_table_graph_results(intensity_tables: Dict[int, IntensityTable],
                                         rounds: Sequence[int],
                                         search_radius: int,
                                         search_radius_max: int,
                                         k_d: float,
                                         anchor_round: int
                                         ) -> IntensityTable:
    """Construct an intensity table from the results of a graph based search of detected spots

    Parameters
    ----------
    intensity_tables : Dict[int, IntensityTable]
        Output from _merge_spots_by_round, contains mapping of intensity tables
        from each round to all the spots detected in them.
    channels, rounds : Sequence[int]
        Channels and rounds present in the ImageStack from which spots were detected.
    search_radius : int
        Euclidean distance in pixels over which to search for spots in subsequent rounds.
    search_radius_max : int
        The maximum (euclidian) distance in pixels allowed between nodes belonging
        to the same sequence
    k_d : float
        distance weight
    anchor_round : int
        The imaging round to seed the search from.

    Returns
    -------
    IntensityTable
        Intensity table from the results of a graph based search of detected spots
    """

    anchor_intensity_table = intensity_tables[anchor_round]
    data = pd.DataFrame()
    for i in intensity_tables:
        data = data.append(
            pd.DataFrame({Axes.X.value: intensity_tables[i][Axes.X.value].values,
                          Axes.Y.value: intensity_tables[i][Axes.Y.value].values,
                          Axes.ZPLANE.value: intensity_tables[i][Axes.ZPLANE.value].values,
                          Axes.CH.value: np.argmax(intensity_tables[i].fillna(
                              0).values, axis=1)[:, i],
                          Axes.ROUND.value: i,
                          'Imax_gf': np.amax(intensity_tables[i].fillna(0).values,
                                             axis=1)[:, i],
                          'p1': intensity_tables[i]['Q'].values,
                          'p0': 1 - intensity_tables[i]['Q'].values,
                          'feature_id': intensity_tables[i].features.values}),
            ignore_index=True)

    res = _runGraphBuilder(data, search_radius, k_d, search_radius_max)
    idx = _baseCalling(res, rounds, search_radius_max)

    # Initialize IntensityTable with anchor round IntensityTable
    intensity_table = anchor_intensity_table.drop('Q')

    # fill IntensityTable
    if len(idx):
        for r in rounds:
            # need numpy indexing to set values in vectorized manner
            intensity_table.values[
                idx[:, anchor_round], :, r] = intensity_tables[r].values[idx[:, r], :, r]

    return IntensityTable(intensity_table)


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
        df = df.data
        df[Axes.CH.value] = np.full(df.shape[0], c)
        round_data[r].append(df)

    # create one dataframe per round
    round_dataframes = {
        k: pd.concat(v, axis=0).reset_index().drop('index', axis=1)
        for k, v in round_data.items()
    }

    return round_dataframes
