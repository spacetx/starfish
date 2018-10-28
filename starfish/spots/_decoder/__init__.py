from typing import Type

import click

from starfish.codebook.codebook import Codebook
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.pipeline import AlgorithmBase, PipelineComponent
from . import _base
from . import per_round_max_channel_decoder


class Decoder(PipelineComponent):

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
    @click.group("decode")
    @click.option("-i", "--input", required=True, type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.option("--codebook", required=True, type=click.Path(exists=True))
    @click.pass_context
    def _cli(ctx, input, output, codebook):
        ctx.obj = dict(
            component=Decoder,
            input=input,
            output=output,
            intensities=IntensityTable.load(input),
            codebook=Codebook.from_json(codebook),
        )


Decoder._cli_register()
