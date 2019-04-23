import platform
from functools import lru_cache
from json import JSONEncoder
from typing import Mapping

import pkg_resources

import starfish.core
from starfish.core.types import CORE_DEPENDENCIES


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
