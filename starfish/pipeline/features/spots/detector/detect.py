from functools import partial
from itertools import product
from typing import Callable, Dict, Tuple, Union, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from starfish.constants import Indices
from starfish.image import ImageStack
from starfish.intensity_table import IntensityTable
from starfish.munge import dataframe_to_multiindex
from starfish.pipeline.features.spot_attributes import SpotAttributes
from starfish.typing import Number


def measure_spot_intensity(
        image: Union[np.ndarray, xr.DataArray],
        spots: pd.DataFrame,
        measurement_function: Callable[[Sequence], Number]) \
        -> pd.Series:
    """measure the intensity of each spot in spots in the corresponding image

    Parameters
    ----------
    image : Union[np.ndarray, xr.DataArray],
        3-d volume in which to measure intensities
    spots : pd.DataFrame
        SpotAttributes table containing coordinates and radii of spots
    measurement_function : Callable[[Sequence], Number]) \
        Function to apply over the spot volumes to identify the intensity (e.g. max, mean, ...)

    Returns
    -------
    pd.Series :
        Intensities for each spot in SpotAttributes

    """

    def fn(row: pd.Series) -> Number:
        int_row: pd.Series = row.astype(int)
        result = measurement_function(
            image[
                int_row['z_min']:int_row['z_max'],
                int_row['y_min']:int_row['y_max'],
                int_row['x_min']:int_row['x_max']
            ]
        )
        return result

    return spots.apply(
        fn,
        axis=1
    )


def measure_spot_intensities(
        data_image: ImageStack,
        spot_attributes: pd.DataFrame,
        measurement_function: Callable[[Sequence], Number], ) \
        -> IntensityTable:
    """given spots found from a reference image, find those spots across a data_image

    Parameters
    ----------
    data_image : ImageStack
        ImageStack containing multiple volumes for which spots' intensities must be calculated
    spot_attributes : pd.Dataframe
        Locations and radii of spots
    measurement_function : Callable[[Sequence], Number]) \
        Function to apply over the spot volumes to identify the intensity (e.g. max, mean, ...)

    Returns
    -------
    IntensityTable :
        3d tensor of (spot, channel, round) information for each coded spot

    """

    n_ch = data_image.shape[Indices.CH]
    n_round = data_image.shape[Indices.ROUND]
    spot_attribute_index = dataframe_to_multiindex(spot_attributes)
    image_shape: Tuple[int, int, int] = data_image.raw_shape[2:]
    intensity_table = IntensityTable.empty_intensity_table(
        spot_attribute_index, n_ch, n_round, image_shape)

    indices = product(range(n_ch), range(n_round))
    for c, r in indices:
        image, _ = data_image.get_slice({Indices.CH: c, Indices.ROUND: r})
        blob_intensities: pd.Series = measure_spot_intensity(
            image, spot_attributes, measurement_function)
        intensity_table[:, c, r] = blob_intensities

    return intensity_table


def concatenate_spot_attributes_to_intensities(
        spot_attributes: Sequence[Tuple[SpotAttributes, Dict[Indices, int]]]) \
        -> IntensityTable:
    """
    Merge multiple spot attributes frames into a single IntensityTable without merging across
    channels and imaging rounds

    Parameters
    ----------
    spot_attributes : Sequence[Tuple[SpotAttributes, Dict[Indices, int]]]
        A sequence of SpotAttribute objects and the Indices (channel, round) that each object is
        associated with.

    Returns
    -------
    IntensityTable :
        concatenated input SpotAttributes, converted to an IntensityTable object

    """
    n_ch: int = max(inds[Indices.CH] for _, inds in spot_attributes) + 1
    n_round: int = max(inds[Indices.ROUND] for _, inds in spot_attributes) + 1

    all_spots = pd.concat([sa.data for sa, inds in spot_attributes])
    spot_attribute_index = dataframe_to_multiindex(all_spots.drop(['spot_id', 'intensity'], axis=1))

    # TODO ambrosejcarr: remove image_shape
    z_max = max(max(sa.data['z']) for sa, inds in spot_attributes)
    y_max = max(max(sa.data['y']) for sa, inds in spot_attributes)
    x_max = max(max(sa.data['x']) for sa, inds in spot_attributes)
    image_shape = (z_max, y_max, x_max)

    intensity_table = IntensityTable.empty_intensity_table(
        spot_attribute_index, n_ch, n_round, image_shape
    )

    i = 0
    for attrs, inds in spot_attributes:
        for _, row in attrs.data.iterrows():
            intensity_table[i, inds[Indices.CH], inds[Indices.ROUND]] = row['intensity']
            i += 1

    return intensity_table


def detect_spots(
        data_image: ImageStack,
        spot_finding_method: Callable[..., SpotAttributes],
        spot_finding_kwargs: Dict=None,
        reference_image: Union[xr.DataArray, np.ndarray]=None,
        reference_from_max_projection: bool=False,
        measurement_function: Callable[[Sequence], Number]=np.max,
) -> IntensityTable:
    """Apply a spot_finding_method to a ImageStack

    Parameters
    ----------
    data_image : ImageStack
        The ImageStack containing spots
    spot_finding_method : Callable[[Any], IntensityTable]
        The method to identify spots
    spot_finding_kwargs : Dict
        additional keyword arguments to pass to spot_finding_method
    reference_image : xr.DataArray
        (Optional) a reference image. If provided, spots will be found in this image, and then
        the locations that correspond to these spots will be measured across each channel and hyb,
        filling in the values in the IntensityTable
    reference_from_max_projection : Tuple[Indices]
        (Optional) if True, create a reference image by max-projecting the channels and imaging
        rounds found in data_image.
    measurement_function : Callable
        the function to apply over the spot area to extract the intensity value (default 'np.max')

    Notes
    -----
    - This class will always detect spots in 3d. If 2d spot detection is desired, the data should
      be projected down to "fake 3d" prior to submission to this function
    - If neither reference_image nor reference_from_max_projection are passed, spots will be
      detected _independently_ in each channel. This assumes a non-multiplex imaging experiment,
      as only one (ch, round) will be measured for each spot.

    Returns
    -------
    IntensityTable :
        IntensityTable containing the intensity of each spot, its radius, and location in pixel
        coordinates

    """

    if spot_finding_kwargs is None:
        spot_finding_kwargs = {}

    if reference_image is not None and reference_from_max_projection:
        raise ValueError(
            'Please pass only one of reference_image and reference_from_max_projection'
        )

    if reference_from_max_projection:
        reference_image = data_image.max_proj(Indices.CH, Indices.ROUND)

    if reference_image is not None:
        reference_spot_locations = spot_finding_method(reference_image, **spot_finding_kwargs)
        intensity_table = measure_spot_intensities(
            data_image=data_image,
            spot_attributes=reference_spot_locations.data,
            measurement_function=measurement_function
        )
    else:
        spot_finding_method = partial(spot_finding_method, **spot_finding_kwargs)
        spot_attributes_list = data_image.transform(
            func=spot_finding_method,
            is_volume=True  # always use volumetric or pseudo-3d (1, n, m) data
        )
        intensity_table = concatenate_spot_attributes_to_intensities(spot_attributes_list)

    return intensity_table
