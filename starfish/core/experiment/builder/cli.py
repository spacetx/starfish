import json

from slicedimage import ImageFormat

from starfish.core.types import Axes
from starfish.core.util import click
from . import write_experiment_json


AUX_IMAGE_NAMES = {
    'nuclei',
    'dots',
}


class StarfishIndex(click.ParamType):

    name = "starfish-index"

    def convert(self, spec_json, param, ctx):
        try:
            spec = json.loads(spec_json)
        except json.decoder.JSONDecodeError:
            self.fail(
                "Could not parse {} into a valid index specification.".format(spec_json))

        return {
            Axes.ROUND: spec.get(Axes.ROUND, 1),
            Axes.CH: spec.get(Axes.CH, 1),
            Axes.ZPLANE: spec.get(Axes.ZPLANE, 1),
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
             Axes.ROUND.value,
             Axes.CH.value,
             Axes.ZPLANE.value))


decorators = [
    click.command(),
    click.argument("output_dir", type=click.Path(exists=True, file_okay=False, writable=True)),
    click.option("--fov-count", type=int, required=True, help="Number of FOVs in this experiment."),
    dimensions_option("primary-image", True),
]
for image_name in AUX_IMAGE_NAMES:
    decorators.append(dimensions_option(image_name, False))


def build(output_dir, fov_count, primary_image_dimensions, **kwargs):
    """generate synthetic experiments"""

    aux_names_to_dimension_cardinality = {
        aux_name: size
        for aux_name, size in kwargs.items()
        if size is not None
    }
    write_experiment_json(
        output_dir, fov_count, ImageFormat.TIFF,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions=aux_names_to_dimension_cardinality,
    )


for decorator in reversed(decorators):
    build = decorator(build)
