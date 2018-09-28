import collections
import json
import os
from typing import Dict, Union


class Config(collections.UserDict):

    def __init__(self, value: Union[str, Dict]=None,
                 key: str="STARFISH_CONFIG"):
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
