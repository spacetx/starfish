from typing import Callable, Optional

from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.types import FunctionSource
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

    Examples
    --------
    Applying a binary opening function.
        >>> from starfish.core.morphology.object.binary_mask.test import factories
        >>> from starfish.morphology import Filter
        >>> from skimage.morphology import disk
        >>> binary_mask_collection = factories.binary_mask_collection_2d()
        >>> opener = Filter.Map("morphology.binary_opening", disk(4))
        >>> opened = opener.run(binary_mask_collection)

    See Also
    --------
    starfish.core.types.Axes

    """

    def __init__(
        self,
            func: str,
            *func_args,
            module: FunctionSource = FunctionSource.np,
            **func_kwargs,
    ) -> None:
        self._func: Callable = module._resolve_method(func)
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
            self._func,
            *self._func_args,
            **self._func_kwargs)
