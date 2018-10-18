import json
from typing import Any, Dict, List, Type

import click

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, PipelineComponent
from . import watershed
from ._base import SegmentationAlgorithmBase


class Segmentation(PipelineComponent):

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return SegmentationAlgorithmBase

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        hyb_stack = ctx.obj["hybridization_stack"]
        nuc_stack = ctx.obj["nuclei_stack"]
        regions = instance.run(hyb_stack, nuc_stack)
        geojson = regions_to_geojson(regions, use_hull=False)

        print("Writing | regions geojson to: {}".format(output))
        with open(output, "w") as f:
            f.write(json.dumps(geojson))


@click.group("segmentation")
@click.option("--hybridization-stack", required=True)  # FIXME: type
@click.option("--nuclei-stack", required=True)  # FIXME: type
@click.option("o", "--output", required=True)
@click.pass_context
def _cli(ctx, hybridization_stack, nuclei_stack, output):
    print('Segmenting ...')
    ctx.obj = dict(
        component=Segmentation,
        output=output,
        hybridization_stack=ImageStack.from_path_or_url(hybridization_stack),
        nuclei_stack=ImageStack.from_path_or_url(nuclei_stack),
    )


def regions_to_geojson(r, use_hull=True) -> List[Dict[str, Dict[str, Any]]]:
    """Convert region geometrical data to geojson format"""

    def make_dict(id_, verts) -> Dict[str, Dict[str, Any]]:
        d = dict()
        c = list(map(lambda x: list(x), list(map(lambda v: [int(v[0]), int(v[1])], verts))))
        d["properties"] = {"id": id_}
        d["geometry"] = {"type": "Polygon", "coordinates": c}
        return d

    if use_hull:
        coordinates = r.hull
    else:
        coordinates = r.coordinates
    return [make_dict(id_, verts) for id_, verts in enumerate(coordinates)]


Segmentation._cli = _cli
Segmentation._cli_register()
