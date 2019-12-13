from typing import Callable, Optional, Tuple, Union

import numpy as np

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.types import FunctionSource, FunctionSourceBundle
from ._base import FilterAlgorithm


class Reduce(FilterAlgorithm):
    """
    Reduce takes masks from one ``BinaryMaskCollection`` and reduces it down to a single mask by
    applying a specified function.  That mask is then returned as a new ``BinaryMaskCollection``.

    An initial value is used to start the reduction process.  The first call to the function will be
    called with ``initial`` and ``M0`` and produce ``R0``.  The second call to the function will be
    called with ``R0`` and ``M1`` and produce ``R1``.

    Parameters
    ----------
    func : Union[str, FunctionSourceBundle]
        Function to reduce the tiles in the input.

        If this value is a string, then the python package is :py:attr:`FunctionSource.np`.

        If this value is a ``FunctionSourceBundle``, then the python package and module name is
        obtained from the bundle.
    initial : Union[np.ndarray, Callable[[Tuple[int, ...]], np.ndarray]]
        An initial array that is the same shape as an uncropped mask, or a callable that accepts the
        shape of an uncropped mask as its parameter and produces an initial array.

    Examples
    --------
    Applying a logical 'AND' across all the masks in a collection.
        >>> from starfish.core.morphology.binary_mask.test import factories
        >>> from starfish.morphology import Filter
        >>> from starfish.types import FunctionSource
        >>> import numpy as np
        >>> from skimage.morphology import disk
        >>> binary_mask_collection = factories.binary_mask_collection_2d()
        >>> initial_mask_producer = lambda shape: np.ones(shape=shape)
        >>> ander = Filter.Reduce(FunctionSource.np("logical_and"), initial_mask_producer)
        >>> anded = anded.run(binary_mask_collection)

    See Also
    --------
    starfish.core.types.Axes

    """

    def __init__(
        self,
            func: Union[str, FunctionSourceBundle],
            initial: Union[np.ndarray, Callable[[Tuple[int, ...]], np.ndarray]],
            *func_args,
            **func_kwargs,
    ) -> None:
        if isinstance(func, str):
            self._func = FunctionSource.np(func)
        elif isinstance(func, FunctionSourceBundle):
            self._func = func
        self._initial = initial
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
        return binary_mask_collection._reduce(
            self._func.resolve(),
            self._initial,
            *self._func_args,
            **self._func_kwargs)
