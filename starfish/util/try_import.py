import functools
from typing import Callable, Optional, Set


def try_import(allowable_module_names: Optional[Set[str]]=None) -> Callable:
    """
    Decorator to apply to a method.  If one of the modules in `allowable_module_names` fail to
    import, raise a friendly error message.  If `allowable_module_names` is None, then all failed
    imports raise a friendly error message.

    Enables large and peripheral dependencies to be excluded from the build.
    """
    def _try_import_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except ImportError as ex:
                module_name = ex.name

                if allowable_module_names is None or module_name in allowable_module_names:
                    raise ImportError(
                        f"{module_name} is an optional dependency of starfish. Please install "
                        f"{module_name} and its dependencies to use this functionality."
                    ) from ex
                else:
                    raise

        return wrapper

    return _try_import_decorator
