from starfish import ImageStack
from starfish.image._filter._base import FilterAlgorithmBase
from starfish.util import click


class SimpleFilterAlgorithm(FilterAlgorithmBase):
    def __init__(self, multiplicand: float):
        self.multiplicand = multiplicand

    def run(self, stack: ImageStack, *args) -> ImageStack:
        numpy_array = stack.xarray
        numpy_array = numpy_array * self.multiplicand
        return ImageStack.from_numpy_array(numpy_array)

    @staticmethod
    @click.command("SimpleFilterAlgorithm")
    @click.option(
        "--multiplicand", default=1.0, type=float)
    @click.pass_context
    def _cli(ctx, multiplicand):
        ctx.obj["component"]._cli_run(ctx, SimpleFilterAlgorithm(multiplicand=multiplicand))


class AdditiveFilterAlgorithm(FilterAlgorithmBase):
    def __init__(self, additive: ImageStack):
        self.additive = additive

    def run(self, stack: ImageStack, *args) -> ImageStack:
        numpy_array = stack.xarray
        numpy_array = numpy_array + stack.xarray
        return ImageStack.from_numpy_array(numpy_array)

    @staticmethod
    @click.command("AdditiveFilterAlgorithm")
    @click.option(
        "--imagestack", type=click.Path(exists=True))
    @click.pass_context
    def _cli(ctx, imagestack):
        ctx.obj["component"]._cli_run(
            ctx,
            AdditiveFilterAlgorithm(additive=ImageStack.from_path_or_url(imagestack)))


class FilterAlgorithmWithMissingConstructorTyping(FilterAlgorithmBase):
    def __init__(self, additive):
        self.additive = additive

    def run(self, stack: ImageStack, *args) -> ImageStack:
        numpy_array = stack.xarray
        numpy_array = numpy_array + stack.xarray
        return ImageStack.from_numpy_array(numpy_array)

    @staticmethod
    @click.command("FilterAlgorithmWithMissingConstructorTyping")
    @click.option(
        "--imagestack", type=click.Path(exists=True))
    @click.pass_context
    def _cli(ctx, imagestack):
        ctx.obj["component"]._cli_run(
            ctx,
            FilterAlgorithmWithMissingConstructorTyping(
                additive=ImageStack.from_path_or_url(imagestack)))


class FilterAlgorithmWithMissingRunTyping(FilterAlgorithmBase):
    def __init__(self, multiplicand: float):
        self.multiplicand = multiplicand

    def run(self, stack, *args) -> ImageStack:
        numpy_array = stack.xarray
        numpy_array = numpy_array * self.multiplicand
        return ImageStack.from_numpy_array(numpy_array)

    @staticmethod
    @click.command("FilterAlgorithmWithMissingRunTyping")
    @click.option(
        "--multiplicand", default=1.0, type=float)
    @click.pass_context
    def _cli(ctx, multiplicand):
        ctx.obj["component"]._cli_run(
            ctx, FilterAlgorithmWithMissingRunTyping(multiplicand=multiplicand))
