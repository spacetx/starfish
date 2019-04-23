from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.measure._regionprops import _RegionProperties
from tqdm import tqdm

from starfish.core.config import StarfishConfig
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.multiprocessing.pool import Pool
from starfish.core.types import Axes, Features, Number, SpotAttributes


class ConnectedComponentDecodingResult(NamedTuple):
    region_properties: List
    label_image: np.ndarray
    decoded_image: np.ndarray


class TargetsMap:

    def __init__(self, targets: np.ndarray) -> None:
        """
        Creates an invertible mapping between string names of Codebook targets and integer IDs
        that can be interpreted by skimage.measure to decode an image.

        Parameters
        ----------
        targets : np.ndarray
            array of string target IDs

        """
        unique_targets = set(targets) - {'nan'}
        sorted_targets = sorted(unique_targets)
        self._int_to_target = dict(zip(range(1, np.iinfo(np.int).max), sorted_targets))
        self._int_to_target[0] = 'nan'
        self._target_to_int = {v: k for (k, v) in self._int_to_target.items()}

    def targets_as_int(self, targets: np.ndarray) -> np.ndarray:
        """Transform an array of targets into their integer representation.

        Parameters
        ----------
        targets : np.ndarray['U']
            array of string targets to be transformed into integer IDs

        Returns
        -------
        np.ndarray[int] :
            array of targets represented by their integer IDs

        """
        return np.array([self._target_to_int[v] for v in targets])

    def targets_as_str(self, targets: np.ndarray) -> np.ndarray:
        """Transform an array of integer IDs into their corresponding string target names.

        Parameters
        ----------
        targets : np.ndarray[int]
            array of int targets to be transformed into string names

        Returns
        -------
        np.ndarray['U']
            array of unicode-encoded target names

        """
        return np.array([self._int_to_target[v] for v in targets])

    def target_as_str(self, integer_target: int) -> np.ndarray:
        return self._int_to_target[integer_target]


class CombineAdjacentFeatures:

    def __init__(
            self,
            min_area: Number,
            max_area: Number,
            connectivity: int=2,
            mask_filtered_features: bool=True
    ) -> None:
        """Combines pixel-wise adjacent features into single larger features using skimage.measure

        Parameters
        ----------
        min_area : Number
            Combined features with area below this value are marked as failing filters
        max_area : Number
            Combined features with area above this value are marked as failing filters
        connectivity : int
            Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. See
            http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label for more
            detail. Default = 2.
        mask_filtered_features : bool
            If True, sets all pixels that are failing filters applied prior to this function equal
            to zero, the background value for skimage.measure.label

        """
        self._min_area = min_area
        self._max_area = max_area
        self._connectivity = connectivity
        self._mask_filtered = mask_filtered_features

    @staticmethod
    def _intensities_to_decoded_image(
            intensities: IntensityTable,
            target_map: TargetsMap,
            mask_filtered_features: bool=True
    ) -> np.ndarray:
        """
        Construct an image where each pixel corresponds to its decoded target, mapped to a unique
        integer ID

        Parameters
        ----------
        intensities : IntensityTable
            Decoded intensities
        target_map : TargetsMap
            Mapping between string target names and integer target IDs
        mask_filtered_features : bool
            If true, all features that fail filters are mapped to zero, which is considered
            'background' and will not decode to a target (default = True).

        Returns
        -------
        np.ndarray[int]
            Image whose pixels are coded as the targets that the ImageStack decoded to at each
            position.

        """
        # reverses the linearization that was used to transform an ImageStack into an IntensityTable
        max_x = intensities[Axes.X.value].values.max() + 1
        max_y = intensities[Axes.Y.value].values.max() + 1
        max_z = intensities[Axes.ZPLANE.value].values.max() + 1

        int_targets = target_map.targets_as_int(intensities[Features.TARGET].values)
        if mask_filtered_features:
            fails_filters = np.where(~intensities[Features.PASSES_THRESHOLDS])[0]
            int_targets[fails_filters] = 0

        decoded_image: np.ndarray = int_targets.reshape((max_z, max_y, max_x))
        return decoded_image

    @staticmethod
    def _calculate_mean_pixel_traces(
            label_image: np.ndarray,
            intensities: IntensityTable,
    ) -> IntensityTable:
        """
        For all pixels that contribute to a connected component, calculate the mean value for
        each (ch, round), producing an average "trace" of a feature across the imaging experiment

        Parameters
        ----------
        label_image : np.ndarray
            An image where all pixels of a connected component share the same integer ID
        intensities : IntensityTable
            decoded intensities

        Returns
        -------
        IntensityTable :
            an IntensityTable where the number of features equals the number of connected components
            and the intensities of each each feature is its mean trace.

        """

        import xarray as xr
        pixel_labels = label_image.reshape(-1)

        # Use a pandas groupby approach-based approach, because it is much faster than xarray

        # If needed, it is possible to be even faster than pandas:
        # https://stackoverflow.com/questions/51975512/\
        # faster-alternative-to-perform-pandas-groupby-operation

        # stack intensities
        stacked = intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))

        # drop into pandas to use their faster groupby
        traces: pd.DataFrame = pd.DataFrame(
            stacked.values,
            index=pixel_labels,
            columns=stacked.traces.to_index()
        )

        #
        distances: pd.Series = pd.Series(
            stacked[Features.DISTANCE].values, index=pixel_labels
        )

        grouped = traces.groupby(level=0)
        pd_mean_pixel_traces = grouped.mean()

        grouped = distances.groupby(level=0)
        pd_mean_distances = grouped.mean()

        pd_xarray = xr.DataArray(
            pd_mean_pixel_traces,
            dims=(Features.AXIS, 'traces'),
            coords=dict(
                traces=('traces', pd_mean_pixel_traces.columns),
                distance=(Features.AXIS, pd_mean_distances),
                features=(Features.AXIS, pd_mean_pixel_traces.index)
            )
        )
        mean_pixel_traces = pd_xarray.unstack('traces')

        # the 0th pixel trace corresponds to background. If present, drop it.
        try:
            mean_pixel_traces = mean_pixel_traces.drop(0, dim=Features.AXIS)
        except KeyError:
            pass

        return mean_pixel_traces

    @staticmethod
    def _single_spot_attributes(
            spot_property: _RegionProperties,
            decoded_image: np.ndarray,
            target_map: TargetsMap,
            min_area: Number,
            max_area: Number,
    ) -> Tuple[Dict[str, int], int]:
        """
        Calculate starfish SpotAttributes from the RegionProperties of a connected component
        feature.

        Parameters
        ----------
        spot_property: _RegionProperties
            Properties of the connected component. Output of skimage.measure.regionprops
        decoded_image : np.ndarray
            Image whose pixels correspond to the targets that the given position in the ImageStack
            decodes to.
        target_map : TargetsMap
            Unique mapping between string target names and int target IDs.
        min_area :
            Combined features with area below this value are marked as failing filters
        max_area : Number
            Combined features with area above this value are marked as failing filters

        Returns
        -------
        Dict[str, Number] :
            spot attribute dictionary for this connected component, containing the x, y, z position,
            target name (str) and feature radius.
        int :
            1 if spot passes size filters, zero otherwise.

        """
        # because of the above skimage issue, we need to support both 2d and 3d properties
        if len(spot_property.centroid) == 3:
            spot_attrs = {
                'z': int(spot_property.centroid[0]),
                'y': int(spot_property.centroid[1]),
                'x': int(spot_property.centroid[2])
            }
        else:  # data is 2d
            spot_attrs = {
                'z': 0,
                'y': int(spot_property.centroid[0]),
                'x': int(spot_property.centroid[1])
            }

        # we're back to 3d or fake-3d here
        target_index = decoded_image[spot_attrs['z'], spot_attrs['y'], spot_attrs['x']]
        spot_attrs[Features.TARGET] = target_map.target_as_str(target_index)
        spot_attrs[Features.SPOT_RADIUS] = spot_property.equivalent_diameter / 2

        # filter intensities for which radius is too small
        passes_area_filter = 1 if min_area <= spot_property.area < max_area else 0
        return spot_attrs, passes_area_filter

    def _create_spot_attributes(
            self,
            region_properties: List[_RegionProperties],
            decoded_image: np.ndarray,
            target_map: TargetsMap,
            n_processes: Optional[int]=None
    ) -> Tuple[SpotAttributes, np.ndarray]:
        """

        Parameters
        ----------
        region_properties : List[_RegionProperties]
            Properties of the each connected component. Output of skimage.measure.regionprops
        decoded_image : np.ndarray
            Image whose pixels correspond to the targets that the given position in the ImageStack
            decodes to.
        target_map : TargetsMap
            Unique mapping between string target names and int target IDs.
        n_processes : Optional[int]=None
            number of processes to devote to measuring spot properties. If None, defaults to the
            result of os.nproc()

        Returns
        -------
        pd.DataFrame :
            DataFrame containing x, y, z, radius, and target name for each connected component
            feature.
        np.ndarray[bool] :
            An array with length equal to the number of features. If zero, indicates that a feature
            has failed area filters.
        """
        pool = Pool(processes=n_processes)
        mapfunc = pool.map
        applyfunc = partial(
            self._single_spot_attributes,
            decoded_image=decoded_image,
            target_map=target_map,
            min_area=self._min_area,
            max_area=self._max_area
        )

        iterable = tqdm(region_properties, disable=(not StarfishConfig().verbose))
        results = mapfunc(applyfunc, iterable)
        spot_attrs, passes_area_filter = zip(*results)

        # update passes filter
        passes_filter = np.array(passes_area_filter, dtype=np.bool)

        spot_attributes = SpotAttributes(pd.DataFrame.from_records(spot_attrs))
        return spot_attributes, passes_filter

    def run(
            self, intensities: IntensityTable,
            n_processes: Optional[int] = None,
    ) -> Tuple[IntensityTable, ConnectedComponentDecodingResult]:
        """
        Execute the combine_adjacent_features method on an IntensityTable containing pixel
        intensities

        Parameters
        ----------
        intensities : IntensityTable
            Pixel intensities of an imaging experiment
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        IntensityTable :
            Table whose features comprise sets of adjacent pixels that decoded to the same target
        ConnectedComponentDecodingResult :
            NamedTuple containing :
                region_properties :
                    the properties of each connected component, in the same order as the
                    IntensityTable
                label_image : np.ndarray
                    An image where all pixels of a connected component share the same integer ID
                decoded_image : np.ndarray
                    Image whose pixels correspond to the targets that the given position in the
                    ImageStack decodes to.

        """

        # map target molecules to integers so they can be reshaped into an image that can
        # be subjected to a connected-component algorithm to find adjacent pixels with the
        # same targets
        targets = intensities[Features.TARGET].values
        target_map = TargetsMap(targets)

        # create the decoded_image
        decoded_image = self._intensities_to_decoded_image(
            intensities,
            target_map,
            self._mask_filtered,
        )

        # label the decoded image to extract connected component features
        label_image: np.ndarray = label(decoded_image, connectivity=self._connectivity)

        # calculate properties of each feature
        props: List = regionprops(np.squeeze(label_image))

        # calculate mean intensities across the pixels of each feature
        mean_pixel_traces = self._calculate_mean_pixel_traces(
            label_image,
            intensities,
        )

        # Create SpotAttributes and determine feature filtering outcomes
        spot_attributes, passes_filter = self._create_spot_attributes(
            props,
            decoded_image,
            target_map,
            n_processes=n_processes
        )

        # augment the SpotAttributes with filtering results and distances from nearest codes
        spot_attributes.data[Features.DISTANCE] = mean_pixel_traces[Features.DISTANCE]
        spot_attributes.data[Features.PASSES_THRESHOLDS] = passes_filter

        # create new indexes for the output IntensityTable
        channel_index = mean_pixel_traces.indexes[Axes.CH]
        round_index = mean_pixel_traces.indexes[Axes.ROUND]
        coords = IntensityTable._build_xarray_coords(spot_attributes, channel_index, round_index)

        # create the output IntensityTable
        dims = (Features.AXIS, Axes.CH.value, Axes.ROUND.value)
        intensity_table = IntensityTable(
            data=mean_pixel_traces, coords=coords, dims=dims
        )

        # combine the various non-IntensityTable results into a NamedTuple before returning
        ccdr = ConnectedComponentDecodingResult(props, label_image, decoded_image)

        return intensity_table, ccdr
