import warnings
from typing import (
    Optional,
    Set,
    Union
)

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.types import Axes, FunctionSource, FunctionSourceBundle, Levels
from ._base import FilterAlgorithm


class Map(FilterAlgorithm):
    """
    Map from input to output by applying a specified function to the input.  The output must have
    the same shape as the input.

    Parameters
    ----------
    func : Union[str, FunctionSourceBundle]
        Function to apply across the dimension(s) specified by ``dims``.

        If this value is a string, then the ``module`` parameter is consulted to determine which
        python package is used to find the function.  If ``module`` is not specified, then the
        default is :py:attr:`FunctionSource.np`.

        If this value is a ``FunctionSourceBundle``, then the python package and module name is
        obtained from the bundle.
    module : Optional[FunctionSource]
        Python module that serves as the source of the function.  It must be listed as one of the
        members of :py:class:`FunctionSource`.

        Currently, the supported FunctionSources are:
        - ``np``: the top-level package of numpy
        - ``scipy``: the top-level package of scipy

        This is being deprecated in favor of specifying the function as a ``FunctionSourceBundle``.
    in_place : bool
        Execute the operation in-place.  (default: False)
    group_by : Set[Axes]
        Axes to split the data along.  For example, splitting a 2D array (axes: X, Y; size: 3, 4)
        by X results in 3 calls to the method, each with arrays of size 4.  (default {Axes.ROUND,
        Axes.CH, Axes.ZPLANE})
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
            func: Union[str, FunctionSourceBundle],
            *func_args,
            module: Optional[FunctionSource] = None,
            in_place: bool = False,
            group_by: Optional[Set[Union[Axes, str]]] = None,
            level_method: Levels = Levels.CLIP,
            **func_kwargs,
    ) -> None:
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
        self.in_place = in_place
        if group_by is None:
            group_by = {Axes.ROUND, Axes.CH, Axes.ZPLANE}
        self.group_by: Set[Axes] = {Axes(axis) for axis in group_by}
        self.level_method = level_method
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
            self.func.resolve(),
            *self.func_args,
            group_by=self.group_by,
            in_place=self.in_place,
            level_method=self.level_method,
            **self.func_kwargs)
