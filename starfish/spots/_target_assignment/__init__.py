import json
import os
from typing import List, Type

import click

from starfish.intensity_table.intensity_table import IntensityTable
from starfish.pipeline import AlgorithmBase, PipelineComponent
from . import point_in_poly
from ._base import TargetAssignmentAlgorithm


class TargetAssignment(PipelineComponent):

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return TargetAssignmentAlgorithm

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        intensity_table = ctx.obj["intensity_table"]
        regions = ctx.obj["regions"]
        intensities = instance.run(intensity_table, regions)
        print("Writing intensities, including cell ids to {}".format(output))
        intensities.save(os.path.join(output))


@click.group("target_assignment")
@click.option("--coordinates-geojson", required=True)  # FIXME: type
@click.option("--intensities", required=True)  # FIXME: type
@click.option("-o", "--output", required=True)
@click.pass_context
def _cli(ctx, coordinates_geojson, intensities, output):

    print('Assigning targets to cells...')
    ctx.obj = dict(
        component=TargetAssignment,
        output=output,
        intensity_table=IntensityTable.load(intensities)
    )

    from starfish import munge

    with open(coordinates_geojson, "r") as fh:
        coordinates = json.load(fh)
    ctx.obj["regions"] = munge.geojson_to_region(coordinates)


TargetAssignment._cli = _cli  # type: ignore
TargetAssignment._cli_register()
