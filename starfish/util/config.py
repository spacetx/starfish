import json
import os
from typing import Any, Dict, Sequence, Union


class NestedDict(dict):

    def __missing__(self, key):
        self[key] = NestedDict()
        return self[key]

    def update(self, source):
        for k, v in source.items():
            if isinstance(v, dict):
                # Doesn't handle recursion
                self[k] = NestedDict()
                self[k].update(v)
            else:
                self[k] = v


class Config(object):

    __NO_VALUE_PASSED = object()

    def __init__(self, value: Union[str, Dict]=None) -> None:
        """
        Parse user arguments, environment variables, and external files to
        generate a nested configuration object.

        Parameters
        ----------
        value: Union[str, Dict]
            Either a json-object-like structure which will be passed unmodified to the
            Config constructor, or a string which will be used to build such a structure. If
            the string starts with an "@", then it will be interpreted as the filename of a
            json file which should be read. Otherwise, it will be parsed as a json string.
        """
        # Record the original values
        self.__value = value

        if not value:
            value = {}

        data = NestedDict()
        if isinstance(value, str):
            if value.startswith("@"):
                filename = os.path.expanduser(value[1:])
                if os.path.exists(filename):
                    with open(filename, "r") as o:
                        data.update(json.loads(o.read()))
            else:
                data.update(json.loads(value))
        else:
            data.update(value)

        self.data = data

    def lookup(self, keys: Sequence[str], value: Any=__NO_VALUE_PASSED) -> Any:
        """
        Parameters
        ----------
        keys: Sequence[str]
            Sequence of keys which will be looked up one after the other.
            If any key does not exist or None is returned, then the provided
            default value will be returned. If no default value is provided,
            an exception is raised.
        value: Any
            Default value to return in the case that the lookup fails. An
            exception will be raised if none is provided.
        """
        data: Any = self.data
        for key in keys:
            try:
                data = data.get(key)
            except (AttributeError, KeyError):
                data = None  # Clear value
                if value is Config.__NO_VALUE_PASSED:
                    raise
        if data:
            # If we've reached here without exception,
            # then we've found the value.
            return data
        elif value is not Config.__NO_VALUE_PASSED:
            return value
        else:
            raise KeyError(keys)
