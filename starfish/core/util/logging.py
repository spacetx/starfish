import platform
from functools import lru_cache
from json import JSONEncoder
from typing import List, Mapping

import pkg_resources

import starfish.core
from starfish.core.types import CORE_DEPENDENCIES, LOG


class Log:
    def __init__(self):
        """
        Class for capturing methods and their parameters applied by an analysis
        pipeline.
        """
        self._log: List[dict] = list()

    def update_log(self, class_instance) -> None:
        """
        Adds a new entry to the log list.

        Parameters
        ----------
        class_instance: The instance of a class being applied to the imagestack
        """
        entry = {"method": class_instance.__class__.__name__,
                 "arguments": class_instance.__dict__,
                 "os": get_os_info(),
                 "dependencies": get_core_dependency_info(),
                 "release tag": get_release_tag(),
                 "starfish version": get_dependency_version('starfish')
                 }
        self._log.append(entry)

    def encode(self):
        return LogEncoder().encode({LOG: self.data})

    @property
    def data(self):
        return self._log


@lru_cache(maxsize=1)
def get_core_dependency_info() -> Mapping[str, str]:
    dependency_info = dict()
    for dependency in CORE_DEPENDENCIES:
        version = get_dependency_version(dependency)
        dependency_info[dependency] = version
    return dependency_info


def get_dependency_version(dependency: str) -> str:
    return pkg_resources.get_distribution(dependency).version


@lru_cache(maxsize=1)
def get_release_tag() -> str:
    if not starfish.core.is_release_tag:
        return "Running starfish from source"
    return starfish.core.is_release_tag


@lru_cache(maxsize=1)
def get_os_info() -> Mapping[str, str]:
    return {"Platform": platform.system(),
            "Version:": platform.version(),
            "Python Version": platform.python_version()}


class LogEncoder(JSONEncoder):
    """
    JSON encodes the List[Dict] pipeline provence log. For simple
    objects use default JSON encoding. For more complex objects
    (ex. Imagestack, Codebook) encode the repr of the object.
    """
    def default(self, o):
        try:
            return super(LogEncoder, self).default(o)
        except TypeError:
            return JSONEncoder().encode(repr(o))
