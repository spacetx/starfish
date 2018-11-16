import click

from starfish.util.click import (
    dimensions_option,
    pass_context_and_log,
)
from . import AUX_IMAGE_NAMES, write_experiment_json



decorators = [
    click.command(),
    click.argument("output_dir", type=click.Path(exists=True, file_okay=False, writable=True)),
    click.option("--fov-count", type=int, required=True, help="Number of FOVs in this experiment."),
    dimensions_option("hybridization", True),
]
for image_name in AUX_IMAGE_NAMES:
    decorators.append(dimensions_option(image_name, False))

@pass_context_and_log
def build(ctx, output_dir, fov_count, hybridization_dimensions, **kwargs):
    write_experiment_json(
        output_dir, fov_count, hybridization_dimensions,
        kwargs
    )

for decorator in reversed(decorators):
    build = decorator(build)
