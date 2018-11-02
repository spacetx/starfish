from typing import Type

import click

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, PipelineComponent
from . import _base
from . import bandpass
from . import clip
from . import gaussian_high_pass
from . import gaussian_low_pass
from . import laplace
from . import mean_high_pass
from . import richardson_lucy_deconvolution
from . import scale_by_percentile
from . import white_tophat
from . import zero_by_channel_magnitude


class Filter(PipelineComponent):

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return _base.FilterAlgorithmBase

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        stack = ctx.obj["stack"]
        filtered = instance.run(stack)
        filtered.write(output)

    @staticmethod
    @click.group("filter")
    @click.option("-i", "--input", type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, input, output):
        print("Filtering images...")
        ctx.obj = dict(
            component=Filter,
            input=input,
            output=output,
            stack=ImageStack.from_path_or_url(input),
        )


Filter._cli_register()
