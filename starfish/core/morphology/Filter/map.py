import warnings
from typing import Optional, Union

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.types import FunctionSource, FunctionSourceBundle
from ._base import FilterAlgorithm


class Map(FilterAlgorithm):
    """
    Map from input to output by applying a specified function to the input.  The output must have
    the same shape as the input.

    Parameters
    ----------
    func : Union[str, FunctionSourceBundle]
        Function to apply across to each of the tiles in the input.

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
    Examples
    --------
    Applying a binary opening function.
        >>> from starfish.core.morphology.binary_mask.test import factories
        >>> from starfish.morphology import Filter
        >>> from starfish.types import FunctionSource
        >>> from skimage.morphology import disk
        >>> binary_mask_collection = factories.binary_mask_collection_2d()
        >>> opener = Filter.Map(FunctionSource.scipy("morphology.binary_opening"), disk(4))
        >>> opened = opener.run(binary_mask_collection)
    """

    def __init__(
        self,
            func: Union[str, FunctionSourceBundle],
            *func_args,
            module: FunctionSource = FunctionSource.np,
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
            self._func = module(func)
        elif isinstance(func, FunctionSourceBundle):
            if module is not None:
                raise ValueError(
                    "When passing in the function as a `FunctionSourceBundle`, module should not "
                    "be set."
                )
            self._func = func
        self._func_args = func_args
        self._func_kwargs = func_kwargs

    def run(
            self,
            binary_mask_collection: BinaryMaskCollection,
            n_processes: Optional[int] = None,
            *args,
            **kwargs
    ) -> BinaryMaskCollection:
        """Map from input to output by applying a specified function to the input.

        Parameters
        ----------
        binary_mask_collection : BinaryMaskCollection
            BinaryMaskCollection to be filtered.
        n_processes : Optional[int]
            The number of processes to use for apply. If None, uses the output of os.cpu_count()
            (default = None).

        Returns
        -------
        BinaryMaskCollection
            Return the results of filter as a new BinaryMaskCollection.
        """

        # Apply the reducing function
        return binary_mask_collection._apply(
            self._func.resolve(),
            *self._func_args,
            **self._func_kwargs)
