import json

import click

from starfish.types import Indices
from . import AUX_IMAGE_NAMES, write_experiment_json


class StarfishIndex(click.ParamType):

    name = "starfish-index"

    def convert(self, spec_json, param, ctx):
        try:
            spec = json.loads(spec_json)
        except json.decoder.JSONDecodeError:
            self.fail(
                "Could not parse {} into a valid index specification.".format(spec_json))

        return {
            Indices.ROUND: spec.get(Indices.ROUND, 1),
            Indices.CH: spec.get(Indices.CH, 1),
            Indices.Z: spec.get(Indices.Z, 1),
        }

def dimensions_option(name, required):
    return click.option(
        "--{}-dimensions".format(name),
        type=StarfishIndex(), required=required,
        help="Dimensions for the {} images.  Should be a json dict, with {}, {}, "
             "and {} as the possible keys.  The value should be the shape along that "
             "dimension.  If a key is not present, the value is assumed to be 0."
             .format(
             name,
             Indices.ROUND.value,
             Indices.CH.value,
             Indices.Z.value))

decorators = [
    click.command(),
    click.argument("output_dir", type=click.Path(exists=True, file_okay=False, writable=True)),
    click.option("--fov-count", type=int, required=True, help="Number of FOVs in this experiment."),
    dimensions_option("hybridization", True),
]
for image_name in AUX_IMAGE_NAMES:
    decorators.append(dimensions_option(image_name, False))

def build(output_dir, fov_count, hybridization_dimensions, **kwargs):
    write_experiment_json(
        output_dir, fov_count, hybridization_dimensions,
        kwargs
    )

for decorator in reversed(decorators):
    build = decorator(build)
