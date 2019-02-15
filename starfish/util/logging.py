import os
import platform
from functools import lru_cache
from json import JSONEncoder
from subprocess import (
    CalledProcessError,
    check_output,
)
from typing import Mapping

import pkg_resources

import starfish
from starfish.types import CORE_DEPENDENCIES


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
def get_git_commit_hash() -> str:
    # First check if in starfish repo
    os.chdir(os.path.dirname(starfish.__file__))
    try:
        check_output(["git", "ls-files", starfish.__file__])
    except CalledProcessError:
        return "Starfish project not under git tracking"
    return check_output(["git", "describe", "--always"]).strip()


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
