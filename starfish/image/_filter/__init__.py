import click
from typing import Type

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
    @click.group(name="filter")
    @click.option("-i", "--input")  # FIXME
    @click.option("o", "--output", required=True)
    @click.pass_context
    def _cli(cls, ctx, input, output):
        print('Filtering images ...')
        ctx.obj = ImageStack.from_path_or_url(input)

for algorithm_cls in Filter._algorithm_to_class_map().values():
    Filter._cli.add_command(algorithm_cls._cli)
