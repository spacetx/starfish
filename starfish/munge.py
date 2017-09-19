import numpy as np
import pandas as pd


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


def scale(stack, metric, clip=False):
    from starfish.stats import stack_describe
    stats = stack_describe(stack)
    ims = stack_to_list(stack)
    res = [im / s[metric] for im, s in zip(ims, stats)]
    return list_to_stack(res)


def gather(df, key, value, cols):
    id_vars = [col for col in df.columns if col not in cols]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt(df, id_vars, id_values, var_name, value_name)


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


def regions_to_geojson(r):
    '''
    Convert region geometrical data to geojson format
    '''

    def make_dict(id, verts):
        d = dict()
        c = list(map(lambda x: list(x), list(map(lambda v: [int(v[0]), int(v[1])], verts))))
        d["properties"] = {"id": id}
        d["geometry"] = {"type": "Polygon", "coordinates": c}
        return d

    return [make_dict(id, verts) for id, verts in enumerate(r.hull)]


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
    # WatershedSegmenter object for now

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
