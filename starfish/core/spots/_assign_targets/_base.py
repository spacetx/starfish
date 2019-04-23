import os
from abc import abstractmethod
from typing import Type

import numpy as np

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.pipeline.pipelinecomponent import PipelineComponent
from starfish.core.segmentation_mask import SegmentationMaskCollection
from starfish.core.util import click


class AssignTargets(PipelineComponent):
    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        intensity_table = ctx.obj["intensity_table"]
        label_image = ctx.obj["label_image"]
        assigned: IntensityTable = instance.run(label_image, intensity_table)
        print(f"Writing intensities, including cell ids to {output}")
        assigned.to_netcdf(os.path.join(output))

    @staticmethod
    @click.group("AssignTargets")
    @click.option("--label-image", required=True, type=click.Path(exists=True))
    @click.option("--intensities", required=True, type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, label_image, intensities, output):
        """assign targets to cells"""

        print('Assigning targets to cells...')
        ctx.obj = dict(
            component=AssignTargets,
            output=output,
            intensity_table=IntensityTable.open_netcdf(intensities),
            label_image=SegmentationMaskCollection.from_disk(label_image)
        )


class AssignTargetsAlgorithm(AlgorithmBase):
    """
    AssignTargets assigns cell IDs to detected spots using an IntensityTable and
    SegmentationMaskCollection.
    """
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return AssignTargets

    @abstractmethod
    def run(
            self,
            label_image: np.ndarray,
            intensity_table: IntensityTable,
            verbose: bool=False,
            in_place: bool=False,
    ) -> IntensityTable:
        """Performs target (e.g. gene) assignment given the spots and the regions."""
        raise NotImplementedError()
