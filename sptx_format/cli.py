from sptx_format import validate_sptx
from starfish.util import click


@click.command()
@click.option("--experiment-json", required=True, metavar="JSON_FILE_OR_URL")
@click.option("--fuzz", is_flag=True)
@click.pass_context
def validate(ctx, experiment_json, fuzz):
    """validate experiment against the jsonschemas"""
    try:
        valid = validate_sptx.validate(experiment_json, fuzz)
        if valid:
            ctx.exit(0)
        else:
            ctx.exit(1)
    except KeyboardInterrupt:
        ctx.exit(3)
