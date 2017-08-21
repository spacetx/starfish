import json

import scipy.misc
from showit import image, tile

from starfish.stats import label_to_regions


def tile_lims(stack, num_std, bar=True, size=20):
    from starfish.stats import stack_describe
    stats = stack_describe(stack)
    lims = [s['mean'] + num_std * s['std'] for s in stats]
    tile(stack, bar=bar, clim=zip([0] * len(lims), lims), size=size)


def image_lims(im, num_std, bar=True, size=20):
    from starfish.stats import im_describe
    stats = im_describe(im)
    lim = stats['mean'] + num_std * stats['std']
    image(im, bar=bar, size=size, clim=[0, lim])


def spots_geo_json(spots_viz_df, fname):
    def make_spot_dict(row):
        row = row[1]
        d = dict()
        d['properties'] = {'id': int(row.spot_id), 'radius': int(row.r)}
        d['geometry'] = {'type': 'Point', 'coordinates': [int(row.x), int(row.y)]}
        return d

    spots_json = [make_spot_dict(row) for row in spots_viz_df.iterrows()]

    with open(fname, 'wb') as outfile:
        json.dump(spots_json, outfile)

    return spots_json


def regions_geo_json(region_labels, fname):
    r = label_to_regions(region_labels)

    def make_region_dict(id, verts):
        d = dict()
        d["properties"] = {"id": id}
        d["geometry"] = {"type": "Polygon",
                         "coordinates": list(map(lambda x: list(x), verts.astype(int)))
                         }
        return d

    regions_json = [make_region_dict(id, verts) for id, verts in enumerate(r.hull)]

    with open(fname, 'wb') as outfile:
        json.dump(regions_json, outfile)
