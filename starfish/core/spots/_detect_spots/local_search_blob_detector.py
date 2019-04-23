"""
Note: this blob detector uses the same underlying methods (skimage.blob_log, skimage.blob_dog) as
the starfish.spots.SpotFinder.BlobDetector. In the future, this and the packages other blob
detectors should be refactored to merge their functionalities.

See: https://github.com/spacetx/starfish/issues/1005
"""

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from click import Choice
from sklearn.neighbors import NearestNeighbors

from starfish.core.compat import blob_dog, blob_log
from starfish.core.image._filter.util import determine_axes_to_group_by
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_from_imagestack_to_intensity_table
from starfish.core.types import Axes, Features, Number, SpotAttributes
from starfish.core.util import click
from ._base import DetectSpotsAlgorithmBase

blob_detectors = {
    'blob_dog': blob_dog,
    'blob_log': blob_log
}


class LocalSearchBlobDetector(DetectSpotsAlgorithmBase):
    """
    Multi-dimensional gaussian spot detector.

    This method is a wrapper for skimage.feature.blob_log that engages in a local radius search
    to match spots up across rounds. In sparse conditions, the search can be seeded in one round
    with a relatively large radius. In crowded images, the spots can be seeded in all rounds and
    consensus filtering can be used to extract codes that are consistently extracted across
    rounds.

    Parameters
    ----------
    min_sigma : float
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    threshold : float
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    detector_method : str ['blob_dog', 'blob_log']
        Name of the type of detection method used from skimage.feature, default: blob_log.
    search_radius : int
        Number of pixels over which to search for spots in other rounds and channels.
    anchor_round : int
        The imaging round against which other rounds will be checked for spots in the same
        approximate pixel location.
    detector_kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the detector_method.

    """

    def __init__(
            self,
            min_sigma: Union[Number, Tuple[Number, ...]],
            max_sigma: Union[Number, Tuple[Number, ...]],
            num_sigma: int,
            threshold: Number,
            detector_method: str='blob_log',
            exclude_border: Optional[int]=None,
            search_radius: int=3,
            anchor_round: int=1,
            **detector_kwargs,
    ) -> None:

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.threshold = threshold
        self.is_volume = True  # TODO test 2-d spot calling
        self.exclude_border = exclude_border
        self.search_radius = search_radius
        self.anchor_round = anchor_round
        self.detector_kwargs = detector_kwargs
        try:
            self.detector_method = blob_detectors[detector_method]
        except ValueError:
            raise ValueError(f"Detector method must be one of {list(blob_detectors.keys())}")

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

        # results = np.ndarray: n_spots x (z, y, x, radius, sigma_z, sigma_y, sigma_x)
        results = self.detector_method(
            data,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            threshold=self.threshold,
            exclude_border=self.exclude_border,
            **self.detector_kwargs
        )

        # if spots were detected
        if results.shape[0]:

            # measure intensities
            z_inds = results[:, 0].astype(int)
            y_inds = results[:, 1].astype(int)
            x_inds = results[:, 2].astype(int)
            intensities = data.values[tuple([z_inds, y_inds, x_inds])]

            # collapse radius if sigma is non-scalar
            if all(np.isscalar(s) for s in (self.min_sigma, self.max_sigma)):
                radius = results[:, 3]
            else:
                radius = np.mean(results[:, -3:], axis=1)

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
        spot_results: Dict[Tuple[int, int], pd.DataFrame]
    ) -> Dict[int, pd.DataFrame]:
        """Merge DataFrames containing spots from different channels into one DataFrame per round.

        Executed on the output of
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
            df[Axes.CH.value] = np.full(df.shape[0], c)
            round_data[r].append(df)

        # create one dataframe per round
        round_dataframes = {
            k: pd.concat(v, axis=0).reset_index().drop('index', axis=1)
            for k, v in round_data.items()
        }

        return round_dataframes

    @staticmethod
    def _match_spots(
        round_dataframes: Dict[int, pd.DataFrame], search_radius: int, anchor_round: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ For each spot in anchor round, find the closest spot within search_radius in all rounds.

        Parameters
        ----------
        round_dataframes : Dict[int, pd.DataFrame]
            Output from _merge_spots_by_round, contains mapping of image volumes from each round to
            all the spots detected in them.
        search_radius : int
            The maximum (euclidean) distance in pixels for a spot to be considered matching in
            a round subsequent to the anchor round.
        anchor_round : int
            The imaging round to seed the local search from.

        Returns
        -------
        pd.DataFrame
            Spots x rounds dataframe containing the distances to the nearest spot. np.nan if
            no spot is detected within search radius
        pd.DataFrame
            Spots x rounds dataframe containing the indices of the nearest spot to the
            corresponding round_dataframe. np.nan if no spot is detected within search radius.

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

            # Build the classifier; chose NN over radius neighbors because data structures are
            # amenable to vectorization, which improves execution time.
            # TODO ambrosejcarr use n_neighbors > 1 to break ties, enable codebook-based finding
            #      use additional axes in dist, ind to retain vectorization.
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(query_coordinates)

            distances, indices = nn.kneighbors(reference_coordinates)
            dist[r] = distances
            ind[r] = indices

        return dist, ind

    @staticmethod
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
        coords = {
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
            intensity_data = round_dataframes[r].loc[spot_indices, 'intensity']
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
        data : ImageStack
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
                "blobs_image shouldn't be set for LocalSearchBlobDetector.  This is likely a usage "
                "error."
            )

        per_tile_spot_results = self._find_spots(
            primary_image, verbose=verbose, n_processes=n_processes)

        per_round_spot_results = self._merge_spots_by_round(per_tile_spot_results)

        distances, indices = self._match_spots(
            per_round_spot_results,
            search_radius=self.search_radius, anchor_round=self.anchor_round
        )

        # TODO implement consensus seeding (SeqFISH)

        intensity_table = self._build_intensity_table(
            per_round_spot_results, distances, indices,
            rounds=primary_image.xarray[Axes.ROUND.value].values,
            channels=primary_image.xarray[Axes.CH.value].values,
            search_radius=self.search_radius,
            anchor_round=self.anchor_round
        )

        transfer_physical_coords_from_imagestack_to_intensity_table(
            image_stack=primary_image, intensity_table=intensity_table
        )

        return intensity_table

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:
        # LocalSearchBlobDetector does not follow the same contract as the remaining spot detectors.
        # TODO: (ambrosejcarr) Rationalize the spot detectors by contract and then remove this hack.
        raise NotImplementedError()

    @staticmethod
    @click.command("LocalSearchBlobDetector")
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
        help="Number of pixels over which to search for spots in other image tiles.")
    @click.pass_context
    def _cli(
        ctx, min_sigma, max_sigma, threshold, overlap, show, detector_method, search_radius
    ) -> None:
        instance = LocalSearchBlobDetector(
            min_sigma, max_sigma, threshold, overlap,
            detector_method=detector_method, search_radius=search_radius
        )
        ctx.obj["component"]._cli_run(ctx, instance)
