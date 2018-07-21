from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd
from skimage.measure import regionprops, label

from starfish.constants import Indices
from starfish.intensity_table import IntensityTable
from starfish.munge import dataframe_to_multiindex
from starfish.typing import Number


class ConnectedComponentDecodingResult(NamedTuple):
    region_properties: List
    label_image: np.ndarray
    decoded_image: np.ndarray


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
        decoded image wherein each target is coded as a different integer. Intended for visualization

    """
    # None needs to map to zero, non-none needs to map to something else.
    int_to_gene = dict(
        zip(range(1, np.iinfo(np.int).max),
            set(intensities.gene_name.values) - {'None'}))
    int_to_gene[0] = 'None'
    gene_to_int = {v: k for (k, v) in int_to_gene.items()}

    # map genes to ints
    gene_list = [gene_to_int[g] for g in intensities.coords[intensities.Constants.GENE.value].values]
    gene_array = np.array(gene_list)

    # reverses the linearization that was used to construct the IntensityTable from the ImageStack
    decoded_image: np.ndarray = gene_array.reshape(intensities.attrs['image_shape'])

    # label each pixel according to its component
    label_image: np.ndarray = label(decoded_image, connectivity=connectivity)

    # calculate the mean value grouped by (ch, round) across all pixels of a connected component
    pixel_labels = label_image.reshape(-1)
    intensities['spot_id'] = (IntensityTable.Constants.FEATURES.value, pixel_labels)
    mean_pixel_traces = intensities.groupby('spot_id').mean(IntensityTable.Constants.FEATURES.value)
    mean_distances = intensities[IntensityTable.Constants.DISTANCE.value].groupby('spot_id').mean(
        IntensityTable.Constants.FEATURES.value)
    mean_pixel_traces[IntensityTable.Constants.DISTANCE.value] = (
        'spot_id',
        np.ravel(mean_distances)
    )

    # "Zero" spot id is background in the label_image. Remove it.
    # TODO ambrosejcarr: can we replace zero with nan above to get around this?
    mean_pixel_traces = mean_pixel_traces.loc[mean_pixel_traces['spot_id'] > 0]
    props: List = regionprops(np.squeeze(label_image))

    # calculate spots and drop ones that fail the area threshold
    spots = []
    passes_filter = np.ones_like(props, dtype=np.bool)  # default is passing (ones)
    for i, spot_property in enumerate(props):
        # TODO ambrosejcarr: add tests for min/max area when fixing masking
        if min_area <= spot_property.area < max_area:

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
            gene_index = decoded_image[spot_attrs['z'], spot_attrs['y'], spot_attrs['x']]
            spot_attrs['gene_name'] = int_to_gene[gene_index]
            spot_attrs['radius'] = spot_property.equivalent_diameter / 2
            spots.append(spot_attrs)
        else:
            passes_filter[i] = 0  # did not pass radius filter

    # filter intensities
    mean_pixel_traces = mean_pixel_traces.loc[passes_filter]

    # now I need to make an IntensityTable from this thing.
    spots_df = pd.DataFrame(spots)
    spots_df[IntensityTable.Constants.DISTANCE.value] = \
        mean_pixel_traces[IntensityTable.Constants.DISTANCE.value]

    # create new indexes for the output IntensityTable
    spots_index = dataframe_to_multiindex(spots_df)
    channel_index = mean_pixel_traces.indexes[Indices.CH]
    round_index = mean_pixel_traces.indexes[Indices.HYB]

    # create the output IntensityTable
    dims = (IntensityTable.Constants.FEATURES.value, Indices.CH.value, Indices.HYB.value)
    attrs = {'image_shape': intensities.attrs['image_shape']}
    intensity_table = IntensityTable(
        data=mean_pixel_traces, coords=(spots_index, channel_index, round_index), dims=dims,
        attrs=attrs
    )

    ccdr = ConnectedComponentDecodingResult(props, label_image, decoded_image)

    return intensity_table, ccdr
