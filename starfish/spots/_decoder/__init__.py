from typing import Type

from starfish.codebook.codebook import Codebook
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.pipeline import AlgorithmBase, import_all_submodules, PipelineComponent
from starfish.util import click
from . import _base
import_all_submodules(__file__, __package__)


COMPONENT_NAME = "decode"


class Decoder(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return _base.DecoderAlgorithmBase

    @classmethod
    def _cli_run(cls, ctx, instance):
        table = ctx.obj["intensities"]
        codes = ctx.obj["codebook"]
        output = ctx.obj["output"]
        intensities = instance.run(table, codes)
        intensities.save(output)

    @staticmethod
    @click.group(COMPONENT_NAME)
    @click.option("-i", "--input", required=True, type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.option("--codebook", required=True, type=click.Path(exists=True))
    @click.pass_context
    def _cli(ctx, input, output, codebook):
        """assign genes to spots"""
        ctx.obj = dict(
            component=Decoder,
            input=input,
            output=output,
            intensities=IntensityTable.load(input),
            codebook=Codebook.from_json(codebook),
        )
