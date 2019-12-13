import importlib
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    cast,
    Mapping,
    Optional,
)


@dataclass
class FunctionSourceBundle:
    """The combination of a ``FunctionSource`` and the method we are looking for."""
    top_level_package: str
    requested_method: str
    actual_method: str

    def resolve(self) -> Callable:
        """Resolve a method.  The method itself might be enclosed in a package, such as
        subpackage.actual_method.  In that case, we will need to attempt to resolve it in the
        following sequence:

        1. import top_level_package, then try to resolve subpackage.actual_method recursively
           through ``getattr`` calls.
        2. import top_level_package.subpackage, then try to resolve actual_method through
           ``gettatr`` calls.

        This is done instead of just creating a bunch of FunctionSource for libraries that have
        a lot of packages that are not implicitly imported by importing the top-level package.
        """
        method_splitted = self.actual_method.split(".")
        splitted = [self.top_level_package]
        splitted.extend(method_splitted)

        for divider in range(1, len(splitted)):
            import_section = splitted[:divider]
            getattr_section = splitted[divider:]

            imported = importlib.import_module(".".join(import_section))

            try:
                for getattr_name in getattr_section:
                    imported = getattr(imported, getattr_name)
                return cast(Callable, imported)
            except AttributeError:
                pass

        raise AttributeError(
            f"Unable to resolve the method {self.requested_method} from package "
            f"{self.top_level_package}")


class FunctionSource(Enum):
    """Each FunctionSource declares a package from which reduction methods can be obtained.
    Generally, the packages should be those that are included as starfish's dependencies for
    reproducibility.

    Many packages are broken into subpackages which are not necessarily implicitly imported when
    importing the top-level package.  For example, ``scipy.linalg`` is not implicitly imported
    when one imports ``scipy``.  To avoid the complexity of enumerating each scipy subpackage in
    FunctionSource, we assemble the fully-qualified method name, and then try all the
    permutations of how one could import that method.

    In the example of ``scipy.linalg.norm``, we try the following:

    1. import ``scipy``, attempt to resolve ``linalg.norm``.
    2. import ``scipy.linalg``, attempt to resolve ``norm``.
    """

    def __init__(self, top_level_package: str, aliases: Optional[Mapping[str, str]] = None):
        self.top_level_package = top_level_package
        self.aliases = aliases or {}

    def __call__(self, method: str) -> FunctionSourceBundle:
        return FunctionSourceBundle(
            self.top_level_package, method, self.aliases.get(method, method))

    np = ("numpy", {'max': 'amax'})
    """Function source for the numpy libraries"""
    scipy = ("scipy",)
    skimage = ("skimage",)
