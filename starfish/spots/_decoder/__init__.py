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
    @click.group("decode")
    @click.option("-i", "--input", required=True)  # FIXME: type
    @click.option("-o", "--output", required=True)
    @click.option("--codebook", required=True)  # FIXME: type
    @click.pass_context
    def _cli(cls, ctx, input, output, codebook):
        ctx.intensities = IntensityTable.load(ctx.input)
        ctx.codebook = Codebook.from_json(ctx.codebook)

    @classmethod
    def _cli_run(clx, ctx, instance):
        intensities = instance.run(ctx.intensities, ctx.codebook)
        intensities.save(ctx.output)

Decoder._cli_register()
