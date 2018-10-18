import click
from sptx_format.validate_sptx import validate as validate_


@click.command()
@click.option("--experiment-json",
              required=True,
              metavar="JSON_FILE_OR_URL")
@click.option("--fuzz", is_flag=True)
@click.pass_context
def validate(ctx, experiment_json, fuzz):
    """invokes validate with the parsed commandline arguments"""
    try:
        valid = validate_(experiment_json, fuzz)
        if valid:
            ctx.exit(0)
        else:
            ctx.exit(1)
    except KeyboardInterrupt:
        ctx.exit(3)
