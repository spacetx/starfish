from abc import abstractmethod
from typing import Type

from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.pipeline.algorithmbase import AlgorithmBase
from starfish.core.pipeline.pipelinecomponent import PipelineComponent
from starfish.core.util import click
from starfish.core.util.click.indirectparams import CodebookParamType


class Decode(PipelineComponent):
    """
    The Decode class exposes methods to compare detected spots to expected patterns of
    fluorescence across the rounds and channels of an experiment and map them to target genes or
    proteins.

    For single molecule FISH or RNAscope experiments, these codebooks are often simple mappings of
    (round, channel) pairs to targets. For coded assays, these codebooks can be much more complex.

    Example codebooks are associated with each experiment in :py:mod:`starfish.data` and can
    be accessed with :py:meth`Experiment.codebook`.
    """
    @classmethod
    def _cli_run(cls, ctx, instance):
        table = ctx.obj["intensities"]
        output = ctx.obj["output"]
        intensities: IntensityTable = instance.run(table)
        intensities.to_netcdf(output)

    @staticmethod
    @click.group("Decode")
    @click.option("-i", "--input", required=True, type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.option("--codebook", required=True, type=CodebookParamType)
    @click.pass_context
    def _cli(ctx, input, output, codebook):
        """assign genes to spots"""
        ctx.obj = dict(
            component=Decode,
            input=input,
            output=output,
            intensities=IntensityTable.open_netcdf(input),
            codebook=codebook,
        )


class DecodeAlgorithmBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return Decode

    @abstractmethod
    def run(self, intensities: IntensityTable, *args):
        """Performs decoding on the spots found, using the codebook specified."""
        raise NotImplementedError()
