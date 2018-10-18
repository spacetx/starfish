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
    @click.group("target_assignment")
    @click.option("--coordinates-geojson", required=True)  # FIXME: type
    @click.option("--intensities", required=True)  # FIXME: type
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(cls, ctx, coordinates_geojson, intensities, output):
        from starfish import munge

        with open(coordinates_geojson, "r") as fh:
            coordinates = json.load(fh)
        ctx.regions = munge.geojson_to_region(coordinates)

        print('Assigning targets to cells...')
        ctx.intensity_table = IntensityTable.load(intensities)

    @classmethod
    def _cli_run(cls, ctx, instance):
        intensities = ctx.instance.run(ctx.intensity_table, ctx.regions)
        print("Writing intensities, including cell ids to {}".format(ctx.output))
        intensities.save(os.path.join(ctx.output))


TargetAssignment._cli_register()
