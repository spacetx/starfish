import click

from sptx_format import validate_sptx
from starfish.util.click import pass_context_and_record


@click.command()
@click.option("--experiment-json", required=True, metavar="JSON_FILE_OR_URL")
@click.option("--fuzz", is_flag=True)
@pass_context_and_record
def validate(ctx, experiment_json, fuzz):
    """invokes validate with the parsed commandline arguments"""
    try:
        valid = validate_sptx.validate(experiment_json, fuzz)
        if valid:
            ctx.exit(0)
        else:
            ctx.exit(1)
    except KeyboardInterrupt:
        ctx.exit(3)
