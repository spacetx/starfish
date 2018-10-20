import json
import os
from typing import Any, Dict, Sequence, Union


class Config(object):

    __NO_VALUE_PASSED = object()

    def __init__(self, value: Union[str, Dict]=None,
                 key: str="STARFISH_CONFIG") -> None:
        """
        Parse user arguments, environment variables, and external files to
        generate a configuration object.

        Parameters
        ----------
        value: Union[str, Dict]
            Either a json-object-like structure which will be passed unmodified to the
            Config constructor, or a string which will be used to build such a structure. If
            the string starts with an "@", then it will be interpreted as the filename of a
            json file which should be read. Otherwise, it will be parsed as a json string.
        key: str
            The name of an environment variable which will be used
            if the value is None. STARFISH_CONFIG will be used if no
            key is provided.
        """
        # Record the original values
        self.__value = value
        self.__key = key

        if value is None:
            value = os.environ.get(key)

        if not value:
            value = {}

        if isinstance(value, str):
            if value.startswith("@"):
                with open(value[1:], "r") as o:
                    data = json.loads(o.read())
            else:
                data = json.loads(value)
        else:
            data = value

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
