from typing import (
    cast,
    Iterable,
    MutableMapping,
    Sequence,
    Union
)

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, Clip, Coordinates, FunctionSource, Number
from starfish.core.util.dtype import preserve_float_range
from ._base import FilterAlgorithm


class Reduce(FilterAlgorithm):
    """
    Reduces the cardinality of one or more axes to 1 by applying a function across those axes.

    Parameters
    ----------
    dims : Iterable[Union[Axes, str]]
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
        >>> from starfish.types import Axes, FunctionSource
        >>> stack = synthetic_stack()
        >>> reducer = Filter.Reduce(
                {Axes.ROUND},
                func="linalg.norm",
                module=FunctionSource.scipy,
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
            func: str = "max",
            module: FunctionSource = FunctionSource.np,
            clip_method: Clip = Clip.CLIP,
            **kwargs
    ) -> None:
        self.dims: Iterable[Axes] = set(Axes(dim) for dim in dims)
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
            Return the results of filter as a new stack.

        """

        # Apply the reducing function
        reduced = stack.xarray.reduce(
            self.func, dim=[dim.value for dim in self.dims], **self.kwargs)

        # Add the reduced dims back and align with the original stack
        reduced = reduced.expand_dims(tuple(dim.value for dim in self.dims))
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
                physical_coords[coord] = cast(Sequence[Number], reduced.coords[coord.value])
        reduced_stack = ImageStack.from_numpy(reduced.values, coordinates=physical_coords)

        return reduced_stack
