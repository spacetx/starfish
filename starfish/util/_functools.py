from typing import Callable, Container, Optional


class StripArguments:
    """
    Class to strip out arguments to a function call.  This is spiritually the opposite of
    functools.partial.  This proxy will remove some arguments and call the nested callable.

    This is used in scenarios where factory methods and lambdas cannot be used, such as dispatching
    work with multiprocessing.  Inputs to multiprocessing must be pickle-able, and factory methods
    and lambdas always generate a lexical scope (i.e., locals()) that cannot be picked.

    This class is initialized with a callable, positional argument positions to remove, and keyword
    arguments to remove.

    For instance, if someone expects a callable that accepts as its parameters (str, int, obj), but
    the callable you have only accepts an int (and does not need the str, obj arguments), then wrap
    the object with _StripArguments(my_callable, positional_arguments_removed=(0, 2)).
    """
    def __init__(
            self,
            func: Callable,
            *,
            positional_arguments_removed: Optional[Container[int]]=None,
            keyword_arguments_removed: Container[set]=None,
    ) -> None:
        self.func = func
        self.positional_arguments_removed = positional_arguments_removed or list()
        self.keyword_arguments_removed = keyword_arguments_removed or set()

    def __call__(self, *args, **kwargs):
        called_args = [
            arg
            for ix, arg in enumerate(args)
            if ix not in self.positional_arguments_removed
        ]
        for keyword_argument_removed in self.keyword_arguments_removed:
            kwargs.remove(keyword_argument_removed)
        return self.func(*called_args, **kwargs)
