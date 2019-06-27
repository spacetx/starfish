import importlib
from enum import Enum
from typing import (
    Callable,
    cast,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union
)

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, Clip, Coordinates, Number
from starfish.core.util import click
from starfish.core.util.dtype import preserve_float_range
from ._base import FilterAlgorithmBase


class Reduce(FilterAlgorithmBase):
    """
    Reduces the cardinality of one or more axes to 1 by applying a function across those axes.

    Parameters
    ----------
    dims : Axes
        one or more Axes to reduce over
    func : str
        Name of a function in the module specified by the ``module`` parameter to apply across the
        dimension(s) specified by dims.  The function is resolved by ``getattr(<module>, func)``,
        except in the cases of predefined aliases.  See :py:class:`FunctionSource` for more
        information about aliases.

        Some common examples for the np FunctionSource:

        - amax: maximum intensity projection (applies np.amax)
        - max: maximum intensity projection (this is an alias for amax and applies np.amax)
        - mean: take the mean across the dim(s) (applies np.mean)
        - sum: sum across the dim(s) (applies np.sum)
    module : FunctionSource
        Python module that serves as the source of the function.  It must be listed as one of the
        members of :py:class:`FunctionSource`.

        Currently, the supported FunctionSources are:
        - ``np``: the top-level package of numpy
        - ``scipy``: the top-level package of scipy
    clip_method : Clip
        (Default Clip.CLIP) Controls the way that data are scaled to retain skimage dtype
        requirements that float data fall in [0, 1].
        Clip.CLIP: data above 1 are set to 1, and below 0 are set to 0
        Clip.SCALE_BY_IMAGE: data above 1 are scaled by the maximum value, with the maximum
        value calculated over the entire ImageStack

    Examples
    --------
    Reducing via max projection.
        >>> from starfish.core.imagestack.test.factories.synthetic_stack import synthetic_stack
        >>> from starfish.image import Filter
        >>> from starfish.types import Axes
        >>> stack = synthetic_stack()
        >>> reducer = Filter.Reduce({Axes.ROUND}, func="max")
        >>> max_proj = reducer.run(stack)

    Reducing via linalg.norm
        >>> from starfish.core.imagestack.test.factories.synthetic_stack import synthetic_stack
        >>> from starfish.image import Filter
        >>> from starfish.types import Axes
        >>> stack = synthetic_stack()
        >>> reducer = Filter.Reduce(
                {Axes.ROUND},
                func="linalg.norm",
                module=Filter.Reduce.FunctionSource.scipy,
                ord=2,
            )
        >>> norm = reducer.run(stack)

    See Also
    --------
    starfish.core.types.Axes

    """

    class FunctionSource(Enum):
        """Each FunctionSource declares a package from which reduction methods can be obtained.
        Generally, the packages should be those that are included as starfish's dependencies for
        reproducibility.

        Many packages are broken into subpackages which are not necessarily implicitly imported when
        importing the top-level package.  For example, ``scipy.linalg`` is not implicitly imported
        when one imports ``scipy``.  To avoid the complexity of enumerating each scipy subpackage in
        FunctionSource, we assemble the fully-qualified method name, and then try all the
        permutations of how one could import that method.

        In the example of ``scipy.linalg.norm``, we try the following:

        1. import ``scipy``, attempt to resolve ``linalg.norm``.
        2. import ``scipy.linalg``, attempt to resolve ``norm``.
        """

        def __init__(self, top_level_package: str, aliases: Optional[Mapping[str, str]] = None):
            self.top_level_package = top_level_package
            self.aliases = aliases or {}

        def _resolve_method(self, method: str) -> Callable:
            """Resolve a method.  The method itself might be enclosed in a package, such as
            subpackage.actual_method.  In that case, we will need to attempt to resolve it in the
            following sequence:

            1. import top_level_package, then try to resolve subpackage.actual_method recursively
               through ``getattr`` calls.
            2. import top_level_package.subpackage, then try to resolve actual_method through
               ``gettatr`` calls.

            This is done instead of just creating a bunch of FunctionSource for libraries that have
            a lot of packages that are not implicitly imported by importing the top-level package.
            """
            # first resolve the aliases.
            actual_method = self.aliases.get(method, method)

            method_splitted = actual_method.split(".")
            splitted = [self.top_level_package]
            splitted.extend(method_splitted)

            for divider in range(1, len(splitted)):
                import_section = splitted[:divider]
                getattr_section = splitted[divider:]

                imported = importlib.import_module(".".join(import_section))

                try:
                    for getattr_name in getattr_section:
                        imported = getattr(imported, getattr_name)
                    return cast(Callable, imported)
                except AttributeError:
                    pass

            raise AttributeError(
                f"Unable to resolve the method {actual_method} from package "
                f"{self.top_level_package}")

        np = ("numpy", {'max': 'amax'})
        """Function source for the numpy libraries"""
        scipy = ("scipy",)

    def __init__(
        self,
            dims: Iterable[Union[Axes, str]],
            func: str = "max",
            module: FunctionSource = FunctionSource.np,
            clip_method: Clip = Clip.CLIP,
            **kwargs
    ) -> None:
        self.dims = dims
        self.func = module._resolve_method(func)
        self.clip_method = clip_method
        self.kwargs = kwargs

    _DEFAULT_TESTING_PARAMETERS = {"dims": ['r'], "func": 'max'}

    def run(
            self,
            stack: ImageStack,
            *args,
    ) -> ImageStack:
        """Performs the dimension reduction with the specifed function

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack. Otherwise return the
            original stack.

        """

        # Apply the reducing function
        reduced = stack._data.reduce(
            self.func, dim=[Axes(dim).value for dim in self.dims], **self.kwargs)

        # Add the reduced dims back and align with the original stack
        reduced = reduced.expand_dims(tuple(Axes(dim).value for dim in self.dims))
        reduced = reduced.transpose(*stack.xarray.dims)

        if self.clip_method == Clip.CLIP:
            reduced = preserve_float_range(reduced, rescale=False)
        else:
            reduced = preserve_float_range(reduced, rescale=True)

        # Update the physical coordinates
        physical_coords: MutableMapping[Coordinates, Sequence[Number]] = {}
        for axis, coord in (
                (Axes.X, Coordinates.X),
                (Axes.Y, Coordinates.Y),
                (Axes.ZPLANE, Coordinates.Z)):
            if axis in self.dims:
                # this axis was projected out of existence.
                assert coord.value not in reduced.coords
                physical_coords[coord] = [np.average(stack._data.coords[coord.value])]
            else:
                physical_coords[coord] = reduced.coords[coord.value]
        reduced_stack = ImageStack.from_numpy(reduced.values, coordinates=physical_coords)

        return reduced_stack

    @staticmethod
    @click.command("Reduce")
    @click.option(
        "--dims",
        type=click.Choice(
            [Axes.ROUND.value, Axes.CH.value, Axes.ZPLANE.value, Axes.X.value, Axes.Y.value]
        ),
        multiple=True,
        help="The dimensions the Imagestack should max project over."
             "For multiple dimensions add multiple --dims. Ex."
             "--dims r --dims c")
    @click.option(
        "--func",
        type=str,
        help="The function to apply across dims."
    )
    @click.option(
        "--module",
        type=click.Choice([member.name for member in list(FunctionSource)]),
        multiple=False,
        help="Module to source the function from.",
        default=FunctionSource.np.name,
    )
    @click.option(
        "--clip-method", default=Clip.CLIP, type=Clip,
        help="method to constrain data to [0,1]. options: 'clip', 'scale_by_image'")
    @click.pass_context
    def _cli(ctx, dims, func, module, clip_method):
        ctx.obj["component"]._cli_run(
            ctx, Reduce(dims, func, Reduce.FunctionSource[module], clip_method))
