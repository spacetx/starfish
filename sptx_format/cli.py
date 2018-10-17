import click
from sptx_format.validate_sptx import validate


@click.command()
@click.option("--experiment-json",
              required=True,
              metavar="JSON_FILE_OR_URL")
@click.option("--fuzz", is_flag=True)
def validate(experiment_json, fuzz):
    """invokes validate with the parsed commandline arguments"""
    try:
        valid = validate(experiment_json, fuzz)
        if valid:
            click.exit(0)
        else:
            click.exit(1)
    except KeyboardInterrupt:
        click.exit(3)
