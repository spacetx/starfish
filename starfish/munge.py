from typing import Any, Dict

import numpy as np
import regional


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


def geojson_to_region(geojson: Dict[Any, Any]) -> regional.many:
    """
    Convert geojson data to region geometrical data.
    """
    def make_region(geometry):
        assert geometry['geometry']['type'] == "Polygon"
        region = [(coordinates[0], coordinates[1])
                  for coordinates in geometry['geometry']['coordinates']]

        return regional.one(region)

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
