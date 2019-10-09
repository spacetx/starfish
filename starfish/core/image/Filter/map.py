from typing import (
    Optional,
    Set,
    Union
)

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, Clip, FunctionSource
from ._base import FilterAlgorithm


class Map(FilterAlgorithm):
    """
    Map from input to output by applying a specified function to the input.  The output must have
    the same shape as the input.

    Parameters
    ----------
    func : str
        Name of a function in the module specified by the ``module`` parameter to apply across the
        dimension(s) specified by dims.  The function is resolved by ``getattr(<module>, func)``,
        except in the cases of predefined aliases.  See :py:class:`FunctionSource` for more
        information about aliases.
    module : FunctionSource
        Python module that serves as the source of the function.  It must be listed as one of the
        members of :py:class:`FunctionSource`.

        Currently, the supported FunctionSources are:
        - ``np``: the top-level package of numpy
        - ``scipy``: the top-level package of scipy
    in_place : bool
        Execute the operation in-place.  (default: False)
    group_by : Set[Axes]
        Axes to split the data along.  For example, splitting a 2D array (axes: X, Y; size: 3, 4)
        by X results in 3 calls to the method, each with arrays of size 4.  (default {Axes.ROUND,
        Axes.CH, Axes.ZPLANE})
    clip_method : Union[str, Clip]
        (Default Clip.CLIP) Controls the way that data are scaled to retain skimage dtype
        requirements that float data fall in [0, 1].
        Clip.CLIP: data above 1 are set to 1, and below 0 are set to 0
        Clip.SCALE_BY_IMAGE: data above 1 are scaled by the maximum value, with the maximum
        value calculated over the entire ImageStack
        Clip.SCALE_BY_CHUNK: data above 1 are scaled by the maximum value, with the maximum
        value calculated over each slice, where slice shapes are determined by the group_by
        parameters.

    Examples
    --------
    Applying a divide function.
        >>> from starfish.core.imagestack.test.factories.synthetic_stack import synthetic_stack
        >>> from starfish.image import Filter
        >>> from starfish.types import Axes
        >>> stack = synthetic_stack()
        >>> divider = Filter.Map("divide", 4)
        >>> by_four = divider.run(stack)

    See Also
    --------
    starfish.core.types.Axes

    """

    def __init__(
        self,
            func: str,
            *func_args,
            module: FunctionSource = FunctionSource.np,
            in_place: bool = False,
            group_by: Optional[Set[Union[Axes, str]]] = None,
            clip_method: Clip = Clip.CLIP,
            **func_kwargs,
    ) -> None:
        self.func = module._resolve_method(func)
        self.in_place = in_place
        if group_by is None:
            group_by = {Axes.ROUND, Axes.CH, Axes.ZPLANE}
        self.group_by: Set[Axes] = {Axes(axis) for axis in group_by}
        self.clip_method = clip_method
        self.func_args = func_args
        self.func_kwargs = func_kwargs

    _DEFAULT_TESTING_PARAMETERS = {"dims": ['r'], "func": 'max'}

    def run(
            self,
            stack: ImageStack,
            *args,
    ) -> Optional[ImageStack]:
        """Map from input to output by applying a specified function to the input.

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.

        Returns
        -------
        Optional[ImageStack] :
            If in-place is False, return the results of filter as a new stack.  Otherwise return
            None

        """

        # Apply the reducing function
        return stack.apply(
            self.func,
            *self.func_args,
            group_by=self.group_by,
            in_place=self.in_place,
            clip_method=self.clip_method,
            **self.func_kwargs)
