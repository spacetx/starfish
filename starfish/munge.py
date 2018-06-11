from typing import Any, Iterable

import numpy as np
import pandas as pd
import regional


def swap(img):
    img_swap = img.swapaxes(0, img.ndim - 1)
    return img_swap


def stack_to_list(stack):
    num_ims = stack.shape[0]
    return [stack[im, :] for im in range(num_ims)]


def list_to_stack(list):
    return np.array(list)


def max_proj(stack):
    im = np.max(stack, axis=0)
    return im


def dataframe_to_multiindex(dataframe: pd.DataFrame) -> pd.MultiIndex:
    """Convert data in a DataFrame to a MultiIndex"""
    names, arrays = zip(*(dataframe.items()))
    return pd.MultiIndex.from_arrays(arrays=arrays, names=names)


def scale(stack, metric, clip=False):
    from starfish.stats import stack_describe
    stats = stack_describe(stack)
    ims = stack_to_list(stack)
    res = [im / s[metric] for im, s in zip(ims, stats)]
    return list_to_stack(res)


def melt(df: pd.DataFrame, new_index_name: Any, new_value_name: Any, melt_columns: Iterable) -> pd.DataFrame:
    """Melt all columns in `melt_columns` into tidy format, tiling all unspecified columns as identifiers

    Columns in `melt_columns` are aggregated into a new column. Their identifiers are stored together in a column
    whose index is `new_index_name`, and their values are stored together in a column whose name is `new_value_name`

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to melt
    new_index_name : Any
        The name of the index column that will be created (typically a str)
    new_value_name : Any
        The name of the column storing the melted values (typically a str)
    melt_columns : Iterable
        Names of columns to be melted into tidy format.

    Examples
    --------
    >>> import pandas as pd
    >>> test = pd.DataFrame(
    ...     data = [
    ...         ['x', '1', '4'],
    ...         ['y', '2', '5'],
    ...         ['z', '3', '6']
    ...     ],
    ...     columns = ['id', 'target_1', 'target_2']
    ... )
    >>> melt(test, 'orig_col', 'melted_values', ['target_1', 'target_2'])
        id  orig_col    melted_values
    0   x   target_1    a
    1   y   target_1    b
    2   z   target_1    c
    3   x   target_2    d
    4   y   target_2    e
    5   z   target_2    f

    Returns
    -------
    pd.DataFrame :
        Melted (tidy) DataFrame

    """
    # any value not passed in melt_columns is retained as an id_var
    id_vars = [col for col in df.columns if col not in melt_columns]

    melted = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=melt_columns,
        var_name=new_index_name,
        value_name=new_value_name
    )

    # melt does not know the dtypes of the new columns, but pandas can normally guess them
    melted = melted.infer_objects()

    return melted


def spots_to_geojson(spots_viz):
    '''
    Convert spot geometrical data to geojson format
    '''

    def make_dict(row):
        row = row[1]
        d = dict()
        d['properties'] = {'id': int(row.spot_id), 'radius': int(row.r)}
        d['geometry'] = {'type': 'Point', 'coordinates': [int(row.x), int(row.y)]}
        return d

    return [make_dict(row) for row in spots_viz.iterrows()]


def geojson_to_region(geojson):
    """
    Convert geojson data to region geometrical data.
    """
    def make_region(geometry):
        assert geometry['geometry']['type'] == "Polygon"

        return regional.one([(coordinates[0], coordinates[1]) for coordinates in geometry['geometry']['coordinates']])

    return regional.many([make_region(geometry) for geometry in geojson])


def relabel(image):
    '''
    This is a local implementation of centrosome.cpmorphology.relabel
    to remove this dependency from starfish.

    Takes a labelled image and relabels each image object consecutively.
    Original code from:
    https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py

    They use a BSD-3 license, which would then have to be propagated to starfish,
    this could be an issue.

    Args
    ----
    image: numpy.ndarray
        A 2d integer array representation of an image with labels

    Returns
    -------
    new_image: numpy.ndarray
        A 2d integer array representation of an image wiht new labels

    n_labels: int
        The number of new unique labels
    '''

    # I've set this as a separate function, rather than binding it to the
    # _WatershedSegmenter object for now

    unique_labels = set(image[image != 0])
    n_labels = len(unique_labels)

    # if the image is unlabelled, return original image
    # warning/message required?
    if n_labels == 0:
        return (image, 0)

    consec_labels = np.arange(n_labels) + 1
    lab_table = np.zeros(max(unique_labels) + 1, int)
    lab_table[[x for x in unique_labels]] = consec_labels

    # Use the label table to remap all of the labels
    new_image = lab_table[image]

    return new_image, n_labels
