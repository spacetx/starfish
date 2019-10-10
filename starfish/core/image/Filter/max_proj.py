import warnings
from typing import Iterable, MutableMapping, Optional, Sequence, Union

import numpy as np

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, Coordinates, Number
from ._base import FilterAlgorithm


class MaxProject(FilterAlgorithm):
    """
    Creates a maximum projection over one or more axis of the image tensor

    .. deprecated:: 0.1.2
        Use `Filter.Reduce(func='max')` instead.

    Parameters
    ----------
    dims : Iterable[Union[Axes, str]]
        one or more Axes to project over

    See Also
    --------
    starfish.types.Axes

    """

    def __init__(self, dims: Iterable[Union[Axes, str]]) -> None:
        warnings.warn(
            "Filter.MaxProject is being deprecated in favor of Filter.Reduce(func='max')",
            DeprecationWarning,
        )
        self.dims = set(Axes(dim) for dim in dims)

    _DEFAULT_TESTING_PARAMETERS = {"dims": 'r'}

    def run(
            self,
            stack: ImageStack,
            verbose: bool = False,
            *args,
    ) -> Optional[ImageStack]:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            if True, report on filtering progress (default = False)
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            The max projection of an image across one or more axis.

        """
        max_projection = stack.xarray.max([dim.value for dim in self.dims])
        max_projection = max_projection.expand_dims(tuple(dim.value for dim in self.dims))
        max_projection = max_projection.transpose(*stack.xarray.dims)
        physical_coords: MutableMapping[Coordinates, Sequence[Number]] = {}
        for axis, coord in (
                (Axes.X, Coordinates.X),
                (Axes.Y, Coordinates.Y),
                (Axes.ZPLANE, Coordinates.Z)):
            if axis in self.dims:
                # this axis was projected out of existence.
                assert coord.value not in max_projection.coords
                physical_coords[coord] = [np.average(stack.xarray.coords[coord.value])]
            else:
                physical_coords[coord] = max_projection.coords[coord.value]
        max_proj_stack = ImageStack.from_numpy(max_projection.values, coordinates=physical_coords)
        return max_proj_stack
