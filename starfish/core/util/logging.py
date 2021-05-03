import json
import platform
from functools import lru_cache
from typing import Any, List, Mapping

import pkg_resources

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

    def _update_log(self, method_name: str, arguments: Any) -> None:
        """
        Adds a new entry to the log list.

        Parameters
        ----------
        method_name : str
            Name of the method triggering this log entry.
        arguments : Any
            Information about the arguments to this method call.
        """
        if isinstance(arguments, dict):
            arguments_as_dict = dict()
            for argument_key, argument_value in arguments.items():
                if hasattr(argument_value, "log") and callable(argument_value.log):
                    try:
                        log = argument_value.log()
                        if isinstance(log, Log):
                            arguments_as_dict[argument_key] = log
                            continue
                    except Exception:
                        pass

                arguments_as_dict[argument_key] = argument_value
            arguments = arguments_as_dict
        elif isinstance(arguments, list):
            arguments_as_list = list()
            for argument_value in arguments:
                if hasattr(argument_value, "log") and callable(argument_value.log):
                    try:
                        log = argument_value.log()
                        if isinstance(log, Log):
                            arguments_as_list.append(log)
                            continue
                    except Exception:
                        pass

                arguments_as_list.append(argument_value)
            arguments = arguments_as_list

        entry = {
            "method": method_name,
            "arguments": arguments,
            "os": get_os_info(),
            "dependencies": get_core_dependency_info(),
            "release tag": get_release_tag(),
            "starfish version": get_dependency_version('starfish')
        }
        self._log.append(entry)

    def update_log(self, class_instance) -> None:
        """
        Adds a new entry to the log list.

        Parameters
        ----------
        class_instance: The instance of a class being applied to the imagestack
        """
        self._update_log(class_instance.__class__.__name__, class_instance.__dict__)

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
