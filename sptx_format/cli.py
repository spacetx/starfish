from time import time

import xarray as xr

from sptx_format import validate_sptx
from starfish.util import click


class DefaultGroup(click.Group):
    """
    Handle the old style invocation by changing --experiment to the subcommand
    """

    def parse_args(self, ctx, args):
        copy = list()
        for arg in args:
            if arg.startswith("--experiment-json"):
                parts = arg.split("=")
                parts[0] = "experiment"
                copy.extend(parts)
            else:
                copy.append(arg)
        super(DefaultGroup, self).parse_args(ctx, copy)


@click.group(cls=DefaultGroup)
@click.option("--experiment-json", required=False, metavar="JSON_FILE_OR_URL")
@click.option("--fuzz", is_flag=True)
@click.pass_context
def validate(ctx, experiment_json, fuzz):
    assert not experiment_json


@validate.command()
@click.argument("experiment_json", metavar="JSON_FILE_OR_URL")
@click.option("--fuzz", is_flag=True)
@click.pass_context
def experiment(ctx, experiment_json, fuzz):
    """validate experiment against the jsonschemas"""

    try:
        valid = validate_sptx.validate(experiment_json, fuzz)
        if valid:
            ctx.exit(0)
        else:
            ctx.exit(1)
    except KeyboardInterrupt:
        ctx.exit(3)


@validate.command()
@click.argument("file")
@click.pass_context
def xarray(ctx, file):
    try:
        d = xr.open_dataset(file)
        print("=" * 60)
        start = time()
        print(d)
        stop = time()
        print("=" * 60)
        print(f"Opened {file} in {stop-start}s.")
        names = set(d.coords._names)
        spots = "z y x radius z_min z_max y_min y_max x_min x_max "
        spots += "intensity spot_id features c r xc yc zc"
        spots = set(spots.split(" "))
        target_spots = set(["cell_id"])
        decoded_spots = set(["target", "distance", "passes_thresholds"])
        if spots.issubset(names):
            if target_spots.issubset(names):
                if decoded_spots.issubset(names):
                    print("Likely decoded spots")
                else:
                    print("Likely target spots")
            else:
                print("Likely just spots")
        else:
            print("Unknown xarray output format!")
            ctx.exit(1)
    except KeyboardInterrupt:
        ctx.exit(3)
    except Exception as e:
        print(f"Invalid xarray: {e}")
        ctx.exit(1)
    else:
        ctx.exit(0)
