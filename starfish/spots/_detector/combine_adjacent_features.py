from functools import partial
from multiprocessing import Pool
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from tqdm import tqdm

from starfish.intensity_table import IntensityTable
from starfish.types import Features, Indices, Number, SpotAttributes


class ConnectedComponentDecodingResult(NamedTuple):
    region_properties: List
    label_image: np.ndarray
    decoded_image: np.ndarray


class TargetsMap:

    def __init__(self, targets: np.ndarray) -> None:
        self._int_to_target = dict(
            zip(range(1, np.iinfo(np.int).max),
                set(targets) - {'nan'}))
        self._int_to_target[0] = 'nan'
        self._target_to_int = {v: k for (k, v) in self._int_to_target.items()}

    def targets_as_int(self, targets: np.ndarray) -> np.ndarray:
        return np.array([self._target_to_int[v] for v in targets])

    def targets_as_str(self, targets: np.ndarray) -> np.ndarray:
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
        self._min_area = min_area
        self._max_area = max_area
        self._connectivity = connectivity
        self._mask_filtered = mask_filtered_features

    @staticmethod
    def _intensities_to_decoded_image(
            intensities: IntensityTable,
            target_map: TargetsMap,
            mask_filtered_features: bool=True
    ):
        # reverses the linearization that was used to transform an ImageStack into an IntensityTable
        max_x = intensities[Indices.X.value].values.max() + 1
        max_y = intensities[Indices.Y.value].values.max() + 1
        max_z = intensities[Indices.Z.value].values.max() + 1

        int_targets = target_map.targets_as_int(intensities[Features.TARGET].values)
        if mask_filtered_features:
            fails_filters = np.where(~intensities[Features.PASSES_FILTERS])[0]
            int_targets[fails_filters] = 0

        decoded_image: np.ndarray = int_targets.reshape((max_z, max_y, max_x))
        return decoded_image

    @staticmethod
    def _calculate_mean_pixel_traces(
            label_image: np.ndarray,
            intensities: IntensityTable,
            passes_filter: pd.Series,
    ):
        pixel_labels = label_image.reshape(-1)
        intensities['spot_id'] = (Features.AXIS, pixel_labels)
        mean_pixel_traces = intensities.groupby('spot_id').mean(Features.AXIS)
        mean_distances = intensities[Features.DISTANCE].groupby('spot_id').mean(Features.AXIS)
        mean_pixel_traces[Features.DISTANCE] = (
            'spot_id',
            np.ravel(mean_distances)
        )

        # the 0th pixel trace corresponds to background. If present, drop it.
        try:
            mean_pixel_traces = mean_pixel_traces.drop(0, dim='spot_id')
        except KeyError:
            pass

        # TODO I think this doesn't do anything, since we drop it above, which would mean we could
        # drop it from this class
        passes_filter[mean_pixel_traces['spot_id'].values == 0] = 0

        return mean_pixel_traces, passes_filter

    @staticmethod
    def _single_spot_attributes(spot_property, decoded_image, target_map, min_area, max_area):
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
            region_properties,
            decoded_image,
            target_map: TargetsMap,
            passes_filter: pd.Series,
            n_processes: Optional[int]=None

    ):
        pool = Pool(n_processes)
        mapfunc = pool.map
        applyfunc = partial(
            self._single_spot_attributes,
            decoded_image=decoded_image,
            target_map=target_map,
            min_area=self._min_area,
            max_area=self._max_area
        )

        iterable = tqdm(region_properties)
        results = mapfunc(applyfunc, iterable)
        spot_attrs, passes_area_filter = zip(*results)

        # update passes filter
        passes_filter = passes_filter.astype(bool) & np.array(passes_area_filter, dtype=np.bool)

        spots_df = pd.DataFrame.from_records(spot_attrs)
        return spots_df, passes_filter

    def run(self, intensities: IntensityTable):

        # map target molecules to integers so they can be reshaped into an image that can
        # be subjected to a connected-component algorithm to find adjacent pixels with the
        # same targets
        targets = intensities[Features.TARGET].values
        target_map = TargetsMap(targets)

        # create the decoded_image, label it, and extract RegionProps (connected components) from it
        decoded_image = self._intensities_to_decoded_image(
            intensities,
            target_map,
            self._mask_filtered,
        )
        label_image: np.ndarray = label(decoded_image, connectivity=self._connectivity)
        props: List = regionprops(np.squeeze(label_image))

        # create a mask to track whether each feature passes filters
        passes_filter_data = np.ones_like(props, dtype=np.bool)  # default is passing (ones)

        # label treats zero as the background value, so starts numbering from 1. To be consistent
        # with other functions, we'll start labeling from zero here (label - 1)
        passes_filter_index = [p.label - 1 for p in props]
        passes_filter = pd.Series(data=passes_filter_data, index=passes_filter_index)

        # calculate mean pixel traces
        mean_pixel_traces, passes_filter = self._calculate_mean_pixel_traces(
            label_image,
            intensities,
            passes_filter,
        )
        assert passes_filter.dtype == np.bool

        # construct a spot attributes table
        spots_df, passes_filter = self._create_spot_attributes(
            props,
            decoded_image,
            target_map,
            passes_filter
        )

        # augment the spot attributes with filtering results and distances from nearest codes
        spots_df[Features.DISTANCE] = mean_pixel_traces[Features.DISTANCE]
        spots_df[Features.PASSES_FILTERS] = passes_filter
        spot_attributes = SpotAttributes(spots_df)

        # create new indexes for the output IntensityTable
        channel_index = mean_pixel_traces.indexes[Indices.CH]
        round_index = mean_pixel_traces.indexes[Indices.ROUND]
        coords = IntensityTable._build_xarray_coords(spot_attributes, channel_index, round_index)

        # create the output IntensityTable
        dims = (Features.AXIS, Indices.CH.value, Indices.ROUND.value)
        intensity_table = IntensityTable(
            data=mean_pixel_traces, coords=coords, dims=dims
        )

        ccdr = ConnectedComponentDecodingResult(props, label_image, decoded_image)

        return intensity_table, ccdr
