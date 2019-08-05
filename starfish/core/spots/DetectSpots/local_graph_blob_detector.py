import itertools
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from click import Choice
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
from skimage import img_as_float
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import h_maxima
from tqdm import tqdm

from starfish.core.compat import blob_dog, blob_log
from starfish.core.image.Filter.util import determine_axes_to_group_by
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_from_imagestack_to_intensity_table
from starfish.core.types import Axes, Features, SpotAttributes
from starfish.core.util import click
from ._base import DetectSpotsAlgorithmBase

detectors = {
    'h_maxima': h_maxima,
    'peak_local_max': peak_local_max,
    'blob_dog': blob_dog,
    'blob_log': blob_log
}


class LocalGraphBlobDetector(DetectSpotsAlgorithmBase):
    """
    Multi-dimensional spot detector.
    This method includes four different spot detectors from skimage and merge the detected
    spots across channels and rounds based on a graphical model.
    implementation.
    Parameters
    ----------
    detector_method : str ['h_maxima', 'peak_local_max', 'blob_dog', 'blob_log']
        Name of the type of detection method used from skimage, default: h_maxima.
    search_radius : int
        Euclidean distance in pixels over which to search for spots in subsequent rounds.
    search_radius_max : int
        The maximum (euclidian) distance in pixels allowed between nodes belonging to the
        same sequence
    detector_kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the detector_method.
    """

    def __init__(
            self,
            detector_method: str='h_maxima',
            search_radius: int=3,
            search_radius_max: int=5,
            **detector_kwargs,
    ) -> None:
        self.is_volume = True  # TODO test 2-d spot calling
        self.search_radius = search_radius
        self.search_radius_max = search_radius_max
        self.anchor_round = 0
        self.detector_kwargs = detector_kwargs
        try:
            self.detector_method = detectors[detector_method]
        except ValueError:
            raise ValueError(f"Detector method must be one of {list(detectors.keys())}")

    def _spot_finder(self, data: xr.DataArray) -> pd.DataFrame:
        """Find spots in a data volume.
        Parameters
        ----------
        data : xr.DataArray
            3D (z, y, x) data volume in which spots will be identified.
        Returns
        -------
        pd.DataFrame
            DataFrame wrapping the output of skimage blob_log or blob_dog with named outputs.
        """

        results = self.detector_method(
            img_as_float(data),
            **self.detector_kwargs
        )

        if (self.detector_method == h_maxima):
            label_h_max = label(results, neighbors=4)
            labels = pd.DataFrame(data={'labels': np.sort(label_h_max[np.where(label_h_max != 0)])})
            # find duplicates labels (=connected components)
            dup = labels.index[labels.duplicated()].tolist()

            # splitting connected regional maxima to get only one local maxima
            max_mask = np.zeros(data.shape)
            max_mask[label_h_max != 0] = 1

            # Compute medoid for connected regional maxima
            for i in range(len(dup)):
                # find coord of points having the same label
                z, r, c = np.where(label_h_max == labels.loc[dup[i], 'labels'])
                meanpoint_x = np.mean(c)
                meanpoint_y = np.mean(r)
                meanpoint_z = np.mean(z)
                dist = [distance.euclidean([meanpoint_z, meanpoint_y, meanpoint_x],
                                           [z[j], r[j], c[j]]) for j in range(len(r))]
                ind = dist.index(min(dist))
                # delete values at ind position.
                z, r, c = np.delete(z, ind), np.delete(r, ind), np.delete(c, ind)
                max_mask[z, r, c] = 0  # set to 0 points != medoid coordinates
            results = max_mask.nonzero()
            results = np.vstack(results).T

        # if spots were detected
        if results.shape[0]:
            # measure intensities
            z_inds = results[:, 0].astype(int)
            y_inds = results[:, 1].astype(int)
            x_inds = results[:, 2].astype(int)
            intensities = data.values[tuple([z_inds, y_inds, x_inds])]

            if (self.detector_method == blob_dog) | (self.detector_method == blob_log):
                # collapse radius if sigma is non-scalar
                if results.shape[1] > 3:
                    radius = np.mean(results[:, -3:], axis=1)
                else:
                    radius = results[:, 3]
            else:
                radius = np.ones(results.shape[0])

            # construct dataframe
            spot_data = pd.DataFrame(
                data={
                    "intensity": intensities,
                    Axes.ZPLANE: z_inds,
                    Axes.Y: y_inds,
                    Axes.X: x_inds,
                    Features.SPOT_RADIUS: radius
                }
            )

        else:
            spot_data = pd.DataFrame(
                data=np.array(
                    [],
                    dtype=[
                        ('intensity', float), ('z', int), ('y', int), ('x', int),
                        (Features.SPOT_RADIUS, float)
                    ]
                )
            )

        return spot_data

    def _find_spots(
        self,
        data_stack: ImageStack,
        verbose: bool=False,
        n_processes: Optional[int]=None
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Find spots in all (z, y, x) volumes of an ImageStack.
        Parameters
        ----------
        data_stack : ImageStack
            Stack containing spots to find.
        Returns
        -------
        Dict[Tuple[int, int], np.ndarray]
            Dictionary mapping (round, channel) pairs to a spot table generated by skimage blob_log
            or blob_dog.
        """
        # find spots in each (r, c) volume
        transform_results = data_stack.transform(
            self._spot_finder,
            group_by=determine_axes_to_group_by(self.is_volume),
            n_processes=n_processes,
        )

        # create output dictionary
        spot_results = {}
        for spot_calls, axes_dict in transform_results:
            r = axes_dict[Axes.ROUND]
            c = axes_dict[Axes.CH]
            spot_results[r, c] = spot_calls

        return spot_results

    @staticmethod
    def _merge_spots_by_round(
        round_dataframes: Dict[int, pd.DataFrame], channels: Sequence[int], rounds: Sequence[int]
    ) -> Dict[int, IntensityTable]:
        """ For each round, find connected components of spots across channels and merge them
        in a single feature.
        Parameters
        ----------
        round_dataframes : Dict[int, pd.DataFrame]
            Output from _merge_spots_by_round, contains mapping of image volumes from each round to
            all the spots detected in them.
        channels, rounds : Sequence[int]
            Channels and rounds present in the ImageStack from which spots were detected.
        Returns
        -------
        Dict[int, pd.DataFrame]
            Dictionary mapping round to the relative IntensityTable.
        """
        intensity_tables = {}

        # get spots matching across channels
        for r, df in round_dataframes.items():
            # Find connected components across channels
            G = nx.Graph()
            kdT = KDTree(df.loc[:, ['z', 'y', 'x']].values)
            pairs = kdT.query_pairs(1, p=1)
            G.add_nodes_from(df.index.values)
            G.add_edges_from(pairs)
            conn_comps = [list(i) for i in nx.connected_components(G)]
            # for each connected component keep detection with highest intensity
            refined_conn_comps = []
            for i in range(len(conn_comps)):
                df_tmp = df.loc[conn_comps[i], :]
                kdT_tmp = KDTree(df_tmp.loc[:, ['z', 'y', 'x']].values)
                # Check if all spots whitin a conn component are at most 1 pixels away
                # from each other (Manhattan distance)
                spot_pairs = len(list(itertools.combinations(np.arange(len(df_tmp)), 2)))
                spots_connected = len(kdT_tmp.query_pairs(2, p=1))  # 2 could be a parameter
                if spot_pairs == spots_connected:
                    # Merge spots
                    refined_conn_comps.append(conn_comps[i])
                else:
                    # split non overlapping signals
                    for j, row in df_tmp.drop_duplicates(['z', 'y', 'x']).iterrows():
                        refined_conn_comps.append(df_tmp[(df_tmp.z == row.z)
                                                  & (df_tmp.y == row.y)
                                                  & (df_tmp.x == row.x)].index.values.tolist())

            data = np.full((len(refined_conn_comps), len(channels), len(rounds)), fill_value=np.nan)
            spot_radius = []
            z = []
            y = []
            x = []
            feature_index = []
            f_idx = 0
            channel_index = []
            round_index = []
            intensity_data = []
            for s in refined_conn_comps:
                df_tmp = df.loc[s]
                anchor_s_idx = df_tmp.intensity.idxmax()
                for i, row in df_tmp.iterrows():
                    data[f_idx, int(row.c), r] = row.intensity

                spot_radius.append(df_tmp.loc[anchor_s_idx, 'radius'])
                z.append(df_tmp.loc[anchor_s_idx, 'z'])
                y.append(df_tmp.loc[anchor_s_idx, 'y'])
                x.append(df_tmp.loc[anchor_s_idx, 'x'])
                feature_index.append(f_idx)
                f_idx += 1
                channel_index.append(df_tmp.loc[anchor_s_idx, 'c'])
                round_index.append(r)
                intensity_data.append(df_tmp.loc[anchor_s_idx, 'intensity'])

            # create IntensityTable
            dims = (Features.AXIS, Axes.CH.value, Axes.ROUND.value)
            coords = {
                Features.SPOT_RADIUS: (Features.AXIS, feature_index),
                Axes.ZPLANE.value: (Features.AXIS, z),
                Axes.Y.value: (Features.AXIS, y),
                Axes.X.value: (Features.AXIS, x),
                Axes.ROUND.value: (Axes.ROUND.value, rounds),
                Axes.CH.value: (Axes.CH.value, channels)
            }
            intensity_table = IntensityTable(data=data, dims=dims, coords=coords)

            intensity_tables[r] = intensity_table

        return intensity_tables

    def _compute_spot_qualities(
            self, data_stack: ImageStack, intensity_tables: Dict[int, IntensityTable]
    ) -> Dict[int, IntensityTable]:
        """Interate over the intesity tables of each round and assign to each feature a quality score
        Parameters
        ----------
        data_stack : ImageStack
            Stack containing spots to find.
        Returns
        -------
        Dict[int,IntensityTable]:
            Dictionary mapping round to the relative IntensityTable with quality coordinate Q
            representing the quality score of each feature.
        -------
        """
        # Fill NaN values of intensity table with intensity values from ImageStack
        for r in intensity_tables:
            features_nan, c_nan = np.where(np.isnan(intensity_tables[r].values[:, :, r]))
            for i in range(len(features_nan)):
                z_nan = int(intensity_tables[r]['z'].values[features_nan[i]])
                y_nan = int(intensity_tables[r]['y'].values[features_nan[i]])
                x_nan = int(intensity_tables[r]['x'].values[features_nan[i]])
                intensity_tables[r].values[features_nan[i],
                                           c_nan[i], r] = data_stack.xarray.values[r,
                                                                                   c_nan[i],
                                                                                   z_nan,
                                                                                   y_nan,
                                                                                   x_nan]
            intensity_tables[r] = intensity_tables[r]

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

    def _prob2Eng(self, p: float) -> float:
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
            self, data: pd.DataFrame,
            l: int, d_th: float,
            k1: float, rounds: np.array,
            dth_max: float) -> Dict:
        """
        Build the graph model for the given connected component and solve the graph
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
        k1 : float
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
                Dvar_tmp = data_tmp.loc[:, ['x', 'y', 'z', 'r', 'ch', 'feature_id']]
                Dvar_tmp['E_0'] = data_tmp.p0.apply(self._prob2Eng)
                Dvar_tmp['E_1'] = data_tmp.p1.apply(self._prob2Eng)
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
                    df1_coords = df1[['x', 'y', 'z']].values
                    df2_coords = df2[['x', 'y', 'z']].values

                    KDTree_h1 = KDTree(df1_coords)
                    KDTree_h2 = KDTree(df2_coords)
                    query = KDTree_h1.query_ball_tree(KDTree_h2, dth_max, p=2)
                    for i in range(len(query)):
                        if len(query[i]):
                            X_idx = [(X_idx_tmp + x) for x in range(len(query[i]))]
                            d = [np.linalg.norm(df1_coords[i] - df2_coords[x]) for x in query[i]]
                            mu_d = [1 / (1 + k1 * x) for x in d]

                            Tvar_tmp = Tvar_tmp.append(
                                pd.DataFrame(data={
                                    'x_idx': X_idx,
                                    'anchestor_x_idx': np.ones(len(query[i])) * df1.index[i],
                                    'descendant_x_idx': df2.index[query[i]].values,
                                    'E_0': [self._prob2Eng(1 - x) for x in mu_d],
                                    'E_1': [self._prob2Eng(x) for x in mu_d],
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

    def _runGraphDecoder(self,
                         data: pd.DataFrame,
                         d_th: float,
                         k1: float,
                         dth_max: float) -> list:
        """
        Find connected components of detected spots across rounds and call the graph
        decoder for each connected component instance.
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe of detected spots with probability values, with columns
            [x, y, z, r, c, idx, p0, p1, feature_id]
        d_th : flaot
             maximum distance inside connected component between two connected spots
        k1 : float
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
            KDTree_h1 = KDTree(data[data.r == h1][['x', 'y', 'z']])
            for h2 in num_hyb[h1:]:
                if h1 != h2:
                    KDTree_h2 = KDTree(data[data.r == h2][['x', 'y', 'z']])
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
                res.append(self._runMaxFlowMinCost(data, int(l), d_th, k1, num_hyb, dth_max))
            # return maxFlowMinCost
            return [x for x in res if x['G'] is not None]
        else:
            return []

    def _baseCalling(self, data: list, rounds: Sequence[int], search_radius_max: int) -> np.ndarray:
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
                        k1 = KDTree(Dvar[['x', 'y', 'z']].values)
                        max_d = np.amax(list(k1.sparse_distance_matrix(k1, np.inf).values()))
                        if max_d <= search_radius_max:
                            idx.append(Dvar[
                                (Dvar.X_idx.isin(c))].sort_values(['r']).feature_id.values)
        return np.array(idx).astype(np.uint)

    def _decode_spots(self,
                      intensity_tables: Dict[int, IntensityTable],
                      channels: Sequence[int],
                      rounds: Sequence[int],
                      search_radius: int,
                      search_radius_max: int,
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
                pd.DataFrame({'x': intensity_tables[i]['x'].values,
                              'y': intensity_tables[i]['y'].values,
                              'z': intensity_tables[i]['z'].values,
                              'ch': np.argmax(intensity_tables[i].fillna(0).values, axis=1)[:, i],
                              'r': i,
                              'Imax_gf': np.amax(intensity_tables[i].fillna(0).values,
                                                 axis=1)[:, i],
                              'p1': intensity_tables[i]['Q'].values,
                              'p0': 1 - intensity_tables[i]['Q'].values,
                              'feature_id': intensity_tables[i].features.values}),
                ignore_index=True)

        res = self._runGraphDecoder(data, search_radius, 0.33, search_radius_max)
        idx = self._baseCalling(res, rounds, search_radius_max)

        # Initialize IntensityTable with anchor round IntensityTable
        intensity_table = anchor_intensity_table.drop('Q')

        # fill IntensityTable
        if len(idx):
            for r in rounds:
                # need numpy indexing to set values in vectorized manner
                intensity_table.values[
                    idx[:, anchor_round], :, r] = intensity_tables[r].values[idx[:, r], :, r]

        return intensity_table

    def run(
            self,
            primary_image: ImageStack,
            blobs_image: Optional[ImageStack] = None,
            blobs_axes: Optional[Tuple[Axes, ...]] = None,
            verbose: bool = False,
            n_processes: Optional[int] = None,
            *args,
    ) -> IntensityTable:
        """Find 1-hot coded spots in data.
        Parameters
        ----------
        primary_image : ImageStack
            Image data containing coded spots.
        verbose : bool
            If True, report on progress of spot finding.
        n_processes : Optional[int]
            Number of processes to devote to spot finding. If None, will use the number of available
            cpus (Default None).
        Notes
        -----
        blobs_image is an unused parameter that is included for testing purposes. It should not
        be passed to this method. If it is passed, the method will trigger a ValueError.
        Returns
        -------
        IntensityTable
            Contains detected coded spots.
        """

        if blobs_image is not None:
            raise ValueError(
                "blobs_image shouldn't be set for LocalGraphBlobDetector.  This is likely a usage "
                "error."
            )

        per_tile_spot_results = self._find_spots(
            primary_image, verbose=verbose, n_processes=n_processes)

        round_data: Mapping[int, List] = defaultdict(list)
        for (r, c), df in per_tile_spot_results.items():
            df[Axes.CH.value] = np.full(df.shape[0], c)
            round_data[r].append(df)

        # create one dataframe per round
        round_dataframes = {
            k: pd.concat(v, axis=0).reset_index().drop('index', axis=1)
            for k, v in round_data.items()
        }

        intensity_tables = self._merge_spots_by_round(
            round_dataframes,
            channels=primary_image.xarray[Axes.CH.value].values,
            rounds=primary_image.xarray[Axes.ROUND.value].values)

        intensity_tables = self._compute_spot_qualities(
            primary_image,
            intensity_tables)

        intensity_table = self._decode_spots(
            intensity_tables=intensity_tables,
            channels=primary_image.xarray[Axes.CH.value].values,
            rounds=primary_image.xarray[Axes.ROUND.value].values,
            search_radius=self.search_radius,
            search_radius_max=self.search_radius_max,
            anchor_round=self.anchor_round)

        # Drop intensities with empty rounds
        drop = [np.any(np.all(np.isnan(intensity_table.values[x, :, :]), axis=0))
                for x in range(intensity_table.shape[0])]
        intensity_table = intensity_table[np.arange(intensity_table.shape[0])[np.invert(drop)]]

        transfer_physical_coords_from_imagestack_to_intensity_table(
            image_stack=primary_image, intensity_table=intensity_table
        )

        return intensity_table

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:
        # LocalGraphBlobDetector does not follow the same contract as the remaining spot detectors.
        # TODO: (ambrosejcarr) Rationalize the spot detectors by contract and then remove this hack.
        raise NotImplementedError()

    @staticmethod
    @click.command("LocalGraphBlobDetector")
    @click.option(
        "--min-sigma", default=4, type=int, help="Minimum spot size (in standard deviation).")
    @click.option(
        "--max-sigma", default=6, type=int, help="Maximum spot size (in standard deviation).")
    @click.option(
        "--threshold", default=.01, type=float, help="Dots threshold.")
    @click.option(
        "--overlap", default=0.5, type=float,
        help="Dots with overlap of greater than this fraction are combined.")
    @click.option(
        "--detector-method", default='blob_log', type=Choice(['blob_log', 'blob_dog']),
        help="Name of the type of the skimage blob detection method.")
    @click.option(
        "--search-radius", default=3, type=int,
        help="Euclidean distance in pixels over which to search for spots in subsequent rounds.")
    @click.option(
        "--search-radius-max", default=5, type=int,
        help="""The maximum (euclidian) distance in pixels allowed between nodes
        belonging to the same sequence.""")
    @click.pass_context
    def _cli(
        ctx, min_sigma, max_sigma, threshold, overlap, show, detector_method,
        search_radius, search_radius_max
    ) -> None:
        instance = LocalGraphBlobDetector(detector_method=detector_method,
                                          search_radius=search_radius,
                                          search_radius_max=search_radius_max,
                                          min_sigma=min_sigma,
                                          max_sigma=max_sigma,
                                          threshold=threshold,
                                          overlap=overlap)
        ctx.obj["component"]._cli_run(ctx, instance)
