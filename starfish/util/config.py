import collections
import json
import os
from typing import Any, Dict, Sequence, Union


class Config(collections.UserDict):

    def __init__(self, value: Union[str, Dict]=None,
                 key: str="STARFISH_CONFIG")->None:
        """
        Parse user-arguments, environment variables, and
        external files to generate a configuration object.

        Parameters
        ----------
        value: Union[str, Dict]
            Either a json-object-like structure which will be passed
            unmodified to the Config constructor, or a string which
            will be used to build such a structure. If the string
            starts with an "@", then it will be interpreted as the
            filename of a json file which should be read. Otherwise,
            it will be parsed as a json string.
        key: str
            The name of an environment variable which will be used
            if the value is None.
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

        collections.UserDict.__init__(self, data)

    def lookup(self, keys: Sequence[str], value: Any=None)->Any:
        data: Any = self.data
        for key in keys:
            try:
                data = data.get(key)
            except Exception:
                return value
        # If we've reached here without exception,
        # then we've found the value.
        if data:
            return data
        return value
