import json
import platform
from functools import lru_cache
from importlib.metadata import version
from typing import List, Mapping

# these are import statements and not from xxx import yyy to break a circular dependency.
import starfish.core
import starfish.core.types


class Log:
    def __init__(self):
        """
        Class for capturing methods and their parameters applied by an analysis
        pipeline.
        """
        self._log: List[dict] = list()

    def update_log(self, class_instance, method_runtime_parameters=None) -> None:
        """
        Adds a new entry to the log list.

        Parameters
        ----------
        class_instance: The instance of a class being applied to the imagestack

        method_runtime_parameters: Any runtime parameters passed to the method
        """
        entry = {"method": class_instance.__class__.__name__,
                 "method_runtime_parameters": method_runtime_parameters,
                 "arguments": class_instance.__dict__,
                 "os": get_os_info(),
                 "dependencies": get_core_dependency_info(),
                 "release tag": get_release_tag(),
                 "starfish version": get_dependency_version('starfish'),
                 }
        self._log.append(entry)

    def encode(self):
        return LogEncoder().encode({starfish.core.types.LOG: self._log})

    @classmethod
    def decode(cls, encoded_log: str):
        log = json.loads(encoded_log)
        log_object = Log()
        log_object._log = log
        return log_object

    @property
    def data(self):
        return self._log


@lru_cache(maxsize=1)
def get_core_dependency_info() -> Mapping[str, str]:
    dependency_info = dict()
    for dependency in starfish.core.types.CORE_DEPENDENCIES:
        version = get_dependency_version(dependency)
        dependency_info[dependency] = version
    return dependency_info


def get_dependency_version(dependency: str) -> str:
    return version(dependency)


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


class LogEncoder(json.JSONEncoder):
    """
    JSON encodes the List[Dict] pipeline provence log. For simple
    objects use default JSON encoding. For more complex objects
    (ex. Imagestack, Codebook) encode the repr of the object.
    """
    def default(self, o):
        try:
            return super(LogEncoder, self).default(o)
        except TypeError:
            return json.JSONEncoder().encode(repr(o))
