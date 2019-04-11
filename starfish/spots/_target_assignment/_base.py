import os
from abc import abstractmethod
from typing import Type

import numpy as np
from skimage.io import imread

from starfish.intensity_table.intensity_table import IntensityTable
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util import click


COMPONENT_NAME = "target_assignment"


class TargetAssignment(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        intensity_table = ctx.obj["intensity_table"]
        label_image = ctx.obj["label_image"]
        assigned = instance.run(label_image, intensity_table)
        print(f"Writing intensities, including cell ids to {output}")
        assigned.save(os.path.join(output))

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("--label-image", required=True, type=click.Path(exists=True))
    @click.option("--intensities", required=True, type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, label_image, intensities, output):
        """assign targets to cells"""

        print('Assigning targets to cells...')
        ctx.obj = dict(
            component=TargetAssignment,
            output=output,
            intensity_table=IntensityTable.load(intensities),
            label_image=imread(label_image)
        )


class TargetAssignmentAlgorithm(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return TargetAssignment

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
