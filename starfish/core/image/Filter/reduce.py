import warnings
from typing import (
    Iterable,
    MutableMapping,
    Optional,
    Union
)

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import (
    ArrayLike,
    Axes,
    Coordinates,
    FunctionSource,
    FunctionSourceBundle,
    Levels,
    Number,
)
from starfish.core.util.levels import levels
from ._base import FilterAlgorithm


class Reduce(FilterAlgorithm):
    """
    Reduces the cardinality of one or more axes to 1 by applying a function across those axes.

    Parameters
    ----------
    dims : Iterable[Union[Axes, str]]
        one or more Axes to reduce over
    func : Union[str, FunctionSourceBundle]
        Function to apply across the dimension(s) specified by ``dims``.

        If this value is a string, then the ``module`` parameter is consulted to determine which
        python package is used to find the function.  If ``module`` is not specified, then the
        default is :py:attr:`FunctionSource.np`.

        If this value is a ``FunctionSourceBundle``, then the python package and module name is
        obtained from the bundle.

        Some common examples for the np FunctionSource:

        - amax: maximum intensity projection (applies np.amax)
        - max: maximum intensity projection (this is an alias for amax and applies np.amax)
        - mean: take the mean across the dim(s) (applies np.mean)
        - sum: sum across the dim(s) (applies np.sum)
    module : Optional[FunctionSource]
        Python module that serves as the source of the function.  It must be listed as one of the
        members of :py:class:`FunctionSource`.

        Currently, the supported FunctionSources are:
        - ``np``: the top-level package of numpy
        - ``scipy``: the top-level package of scipy

        This is being deprecated in favor of specifying the function as a ``FunctionSourceBundle``.
    clip_method : Optional[Union[str, :py:class:`~starfish.types.Clip`]]
        Deprecated method to control the way that data are scaled to retain skimage dtype
        requirements that float data fall in [0, 1].  In all modes, data below 0 are set to 0.

        - Clip.CLIP: data above 1 are set to 1.  This has been replaced with
          level_method=Levels.CLIP.
        - Clip.SCALE_BY_IMAGE: when any data in the entire ImageStack is greater than 1, the entire
          ImageStack is scaled by the maximum value in the ImageStack.  This has been replaced with
          level_method=Levels.SCALE_SATURATED_BY_IMAGE.
        - Clip.SCALE_BY_CHUNK: when any data in any slice is greater than 1, each slice is scaled by
          the maximum value found in that slice.  The slice shapes are determined by the
          ``group_by`` parameters.  This has been replaced with
          level_method=Levels.SCALE_SATURATED_BY_CHUNK.
    level_method : :py:class:`~starfish.types.Levels`
        Controls the way that data are scaled to retain skimage dtype requirements that float data
        fall in [0, 1].  In all modes, data below 0 are set to 0.

        - Levels.CLIP (default): data above 1 are set to 1.
        - Levels.SCALE_SATURATED_BY_IMAGE: when any data in the entire ImageStack is greater
          than 1, the entire ImageStack is scaled by the maximum value in the ImageStack.
        - Levels.SCALE_SATURATED_BY_CHUNK: when any data in any slice is greater than 1, each
          slice is scaled by the maximum value found in that slice.  The slice shapes are
          determined by the ``group_by`` parameters.
        - Levels.SCALE_BY_IMAGE: scale the entire ImageStack by the maximum value in the
          ImageStack.
        - Levels.SCALE_BY_CHUNK: scale each slice by the maximum value found in that slice.  The
          slice shapes are determined by the ``group_by`` parameters.

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
        >>> from starfish.types import Axes, FunctionSource
        >>> stack = synthetic_stack()
        >>> reducer = Filter.Reduce(
                {Axes.ROUND},
                func=FunctionSource.scipy("linalg.norm"),
                ord=2,
            )
        >>> norm = reducer.run(stack)

    See Also
    --------
    starfish.core.types.Axes

    """

    def __init__(
        self,
            dims: Iterable[Union[Axes, str]],
            func: Union[str, FunctionSourceBundle] = "max",
            module: Optional[FunctionSource] = None,
            level_method: Levels = Levels.CLIP,
            **kwargs
    ) -> None:
        self.dims: Iterable[Axes] = set(Axes(dim) for dim in dims)
        if isinstance(func, str):
            if module is not None:
                warnings.warn(
                    f"The module parameter is being deprecated.  Use "
                    f"`func=FunctionSource.{module.name}{func} instead.",
                    DeprecationWarning)
            else:
                module = FunctionSource.np
            self.func = module(func)
        elif isinstance(func, FunctionSourceBundle):
            if module is not None:
                raise ValueError(
                    "When passing in the function as a `FunctionSourceBundle`, module should not "
                    "be set."
                )
            self.func = func
        self.level_method = level_method
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
            Return the results of filter as a new stack.

        """

        # Apply the reducing function
        reduced = stack.xarray.reduce(
            self.func.resolve(), dim=[dim.value for dim in self.dims], **self.kwargs)

        # Add the reduced dims back and align with the original stack
        reduced = reduced.expand_dims(tuple(dim.value for dim in self.dims))
        reduced = reduced.transpose(*stack.xarray.dims)

        if self.level_method == Levels.CLIP:
            reduced = levels(reduced)
        elif self.level_method in (Levels.SCALE_BY_CHUNK, Levels.SCALE_BY_IMAGE):
            reduced = levels(reduced, rescale=True)
        elif self.level_method in (
                Levels.SCALE_SATURATED_BY_CHUNK, Levels.SCALE_SATURATED_BY_IMAGE):
            reduced = levels(reduced, rescale_saturated=True)

        # Update the physical coordinates
        physical_coords: MutableMapping[Coordinates, ArrayLike[Number]] = {}
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
