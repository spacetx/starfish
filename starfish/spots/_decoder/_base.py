from abc import abstractmethod
from typing import Type

from starfish.codebook.codebook import Codebook
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent
from starfish.util import click
from starfish.util.click.indirectparams import CodebookParamType

COMPONENT_NAME = "decode"


class Decoder(PipelineComponent):
    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

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
    @click.option("--codebook", required=True, type=CodebookParamType)
    @click.pass_context
    def _cli(ctx, input, output, codebook):
        """assign genes to spots"""
        ctx.obj = dict(
            component=Decoder,
            input=input,
            output=output,
            intensities=IntensityTable.load(input),
            codebook=codebook,
        )


class DecoderAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return Decoder

    @abstractmethod
    def run(self, intensities: IntensityTable, codebook: Codebook, *args):
        """Performs decoding on the spots found, using the codebook specified."""
        raise NotImplementedError()
