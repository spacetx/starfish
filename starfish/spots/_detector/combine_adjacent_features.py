from typing import List, NamedTuple, Tuple, Optional
from multiprocessing import Pool
from functools import partial

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

    def __init__(self, targets: np.ndarray):
        self._int_to_target = dict(
            zip(range(1, np.iinfo(np.int).max),
                set(targets) - {'nan'}))
        self._int_to_target[0] = 'nan'
        self._target_to_int = {v: k for (k, v) in self._int_to_target.items()}

    def targets_as_int(self, targets: np.ndarray):
        return np.array([self._target_to_int[v] for v in targets])

    def targets_as_str(self, targets: np.ndarray):
        return np.array([self._int_to_target[v] for v in targets])

    def target_as_str(self, integer_target: int):
        return self._int_to_target[integer_target]


class CombineAdjacentFeatures:
    def __init__(self, min_area: Number, max_area: Number, connectivity: int=2):
        self._min_area = min_area
        self._max_area = max_area
        self._connectivity = connectivity

    @staticmethod
    def _intensities_to_decoded_image(intensities: IntensityTable, target_map: TargetsMap):
        # reverses the linearization that was used to transform an ImageStack into an IntensityTable
        max_x = intensities[Indices.X.value].values.max() + 1
        max_y = intensities[Indices.Y.value].values.max() + 1
        max_z = intensities[Indices.Z.value].values.max() + 1
        int_targets = target_map.targets_as_int(intensities[Features.TARGET].values)
        decoded_image: np.ndarray = int_targets.reshape((max_z, max_y, max_x))
        return decoded_image

    @staticmethod
    def _calculate_mean_pixel_traces(
            label_image: np.ndarray,
            intensities: IntensityTable,
            passes_filter: np.ndarray,
    ):
        pixel_labels = label_image.reshape(-1)
        intensities['spot_id'] = (Features.AXIS, pixel_labels)
        mean_pixel_traces = intensities.groupby('spot_id').mean(Features.AXIS)
        mean_distances = intensities[Features.DISTANCE].groupby('spot_id').mean(Features.AXIS)
        mean_pixel_traces[Features.DISTANCE] = (
            'spot_id',
            np.ravel(mean_distances)
        )

        passes_filter[mean_pixel_traces['spot_id'] == 0] = 0

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
        passes_filter = 1 if min_area <= spot_property.area < max_area else 0
        return spot_attrs, passes_filter

    def _create_spot_attributes(
            self,
            region_properties,
            decoded_image,
            target_map: TargetsMap,
            passes_filter: np.ndarray,
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
        spot_attrs, passes_filter = zip(*results)

        # spots = []
        # for i, spot_property in enumerate(region_properties):
        #
        #     # because of the above skimage issue, we need to support both 2d and 3d properties
        #     if len(spot_property.centroid) == 3:
        #         spot_attrs = {
        #             'z': int(spot_property.centroid[0]),
        #             'y': int(spot_property.centroid[1]),
        #             'x': int(spot_property.centroid[2])
        #         }
        #     else:  # data is 2d
        #         spot_attrs = {
        #             'z': 0,
        #             'y': int(spot_property.centroid[0]),
        #             'x': int(spot_property.centroid[1])
        #         }
        #
        #     # we're back to 3d or fake-3d here
        #     target_index = decoded_image[spot_attrs['z'], spot_attrs['y'], spot_attrs['x']]
        #     spot_attrs[Features.TARGET] = target_map.target_as_str(target_index)
        #     spot_attrs[Features.SPOT_RADIUS] = spot_property.equivalent_diameter / 2
        #     spots.append(spot_attrs)
        #
        #     # filter intensities for which radius is too small
        #     if not self._min_area <= spot_property.area < self._max_area:
        #         passes_filter[i] = 0  # did not pass radius filter

        spots_df = pd.DataFrame.from_records(spot_attrs)
        return spots_df, np.array(passes_filter)

    def run(self, intensities: IntensityTable):

        # map target molecules to integers so they can be reshaped into an image that can
        # be subjected to a connected-component algorithm to find adjacent pixels with the
        # same targets
        targets = intensities[Features.TARGET].values
        target_map = TargetsMap(targets)

        # create the decoded_image, label it, and extract RegionProps (connected components) from it
        decoded_image = self._intensities_to_decoded_image(intensities, target_map)
        label_image: np.ndarray = label(decoded_image, connectivity=self._connectivity)
        props: List = regionprops(np.squeeze(label_image))
        passes_filter = np.ones_like(props, dtype=np.bool)  # default is passing (ones)

        # calculate mean pixel traces
        mean_pixel_traces, passes_filter = self._calculate_mean_pixel_traces(
            label_image,
            intensities,
            passes_filter
        )

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


def combine_adjacent_features(
        intensities: IntensityTable, min_area: Number, max_area: Number, connectivity: int=2,
) -> Tuple[IntensityTable, ConnectedComponentDecodingResult]:
    """

    Parameters
    ----------
    intensities : IntensityTable
        pixel-valued IntensityTable
    min_area : Number
        discard spots whose total pixel area is less than this value
    max_area : Number
        discard spots whose total pixel area is greater than this value
    connectivity : int
        the connectivity allowed when constructing connected components. Intuitively it can be
        thought of as the number of square steps allowed between pixels that are considered
        connected. 1 enforces adjacent pixels. 2 (default) allows diagonal pixels in the same
        plane. 3 allows diagonal pixels in 3d.

    Returns
    -------
    IntensityTable :
        spot intensities
    ConnectedComponentDecodingResult :
        region properties for each spot (see skimage.measure.regionprops)
        label image wherein each connected component (spot) is coded with a different integer
        decoded image wherein each target is coded as a different integer. Intended for
        visualization

    """
    # None needs to map to zero, non-none needs to map to something else.
    int_to_target = dict(
        zip(range(1, np.iinfo(np.int).max),
            set(intensities[Features.AXIS][Features.TARGET].values) - {'nan'}))
    int_to_target[0] = 'nan'
    target_to_int = {v: k for (k, v) in int_to_target.items()}

    # map targets to ints
    target_list = [
        target_to_int[g] for g in intensities.coords[Features.TARGET].values]
    target_array = np.array(target_list)

    # reverses the linearization that was used to construct the IntensityTable from the ImageStack
    max_x = intensities[Indices.X.value].values.max() + 1
    max_y = intensities[Indices.Y.value].values.max() + 1
    max_z = intensities[Indices.Z.value].values.max() + 1
    decoded_image: np.ndarray = target_array.reshape((max_z, max_y, max_x))

    # label each pixel according to its component
    label_image: np.ndarray = label(decoded_image, connectivity=connectivity)

    # calculate the mean value grouped by (ch, round) across all pixels of a connected component
    SPOT_ID = 'spot_id'
    pixel_labels = label_image.reshape(-1)
    intensities[SPOT_ID] = (Features.AXIS, pixel_labels)
    mean_pixel_traces = intensities.groupby(SPOT_ID).mean(Features.AXIS)
    mean_distances = intensities[Features.DISTANCE].groupby(SPOT_ID).mean(Features.AXIS)
    mean_pixel_traces[Features.DISTANCE] = (
        SPOT_ID,
        np.ravel(mean_distances)
    )

    props: List = regionprops(np.squeeze(label_image))
    passes_filter = np.ones_like(props, dtype=np.bool)  # default is passing (ones)

    # filter traces that are too distant from nearest code (0 in label image)
    passes_filter[mean_pixel_traces[SPOT_ID] == 0] = 0

    # TODO this could be multiprocessed to speed up single-core analysis
    # calculate properties for the spot_data
    spots = []
    for i, spot_property in enumerate(props):

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
        spot_attrs[Features.TARGET] = int_to_target[target_index]
        spot_attrs[Features.SPOT_RADIUS] = spot_property.equivalent_diameter / 2
        spots.append(spot_attrs)

        # filter intensities for which radius is too small
        if not min_area <= spot_property.area < max_area:
            passes_filter[i] = 0  # did not pass radius filter

    # create an IntensityTable
    spots_df = pd.DataFrame(spots)
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
