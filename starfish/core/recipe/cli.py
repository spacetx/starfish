from slicedimage.io import resolve_path_or_url

from starfish.core.config import StarfishConfig
from starfish.core.starfish import starfish
from starfish.core.util import click
from .recipe import Recipe


@starfish.command("recipe")
@click.option("--recipe", required=True, type=str, metavar="RECIPE_PATH_OR_URL")
@click.option(
    "--input", type=str, multiple=True, metavar="INPUT_FILE_PATH_OR_URL",
    help="input file paths or urls to map to the recipe input parameters")
@click.option(
    "--output", type=str, multiple=True, metavar="OUTPUT_FILE_PATH",
    help="output file paths to write recipe outputs to")
@click.pass_context
def run_recipe(ctx, recipe, input, output):
    """Runs a recipe with a given set of inputs and outputs."""
    config = StarfishConfig()

    backend, relativeurl, _ = resolve_path_or_url(
        recipe, backend_config=config.slicedimage)
    with backend.read_contextmanager(relativeurl) as fh:
        recipe_str = fh.read()

    recipe_obj = Recipe(recipe_str, input, output)
    recipe_obj.run_and_save()
