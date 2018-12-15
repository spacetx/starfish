import os

from starfish.util.config import Config, NestedDict


class environ(object):
    """Overrides environment variables (prefixed with ``STARFISH_``)
    for the duration of the call.

    Examples
    --------
    Turn on validation:

        >>> from starfish.config import environ
        >>> from starfish.experiment.experiment import Experiment
        >>> with environ(VALIDATION_STRICT="true"):
        >>>     Experiment.from_json(URL)

    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.orig = dict()
        for k, newval in self.kwargs.items():
            if not k.startswith("STARFISH_"):
                k = "STARFISH_%s" % k
            old = os.environ.get(k, None)
            try:
                os.environ[k] = newval
                self.orig[k] = old  # Only store if successful
            except TypeError:
                raise
        return self

    def __exit__(self, *args):
        for k, oldval in self.orig.items():
            if oldval is None:
                del os.environ[k]
            else:
                os.environ[k] = oldval


class StarfishConfig(object):
    """
    Application specific configuration settings which can be loaded throughout
    the starfish codebase.

    Attributes
    ----------
    slicedimage : dictionary
        Subdictionary that can be passed to slicedimage.io methods.
    strict : bool
        Whether or not loaded json should be validated.
    verbose : bool
        Controls output like from tqdm

    Examples
    --------
    Check strict property

        >>> from starfish.config import StarfishConfig
        >>> config = StarfishConfig()
        >>> if config.strict:
        >>>     validate(json)

    Default starfish configuration equivalent:

        >>> {
        >>>     "slicedimage": {
        >>>         "caching": {
        >>>             "debug": false,
        >>>             "directory": "~/.starfish/cache",
        >>>             "size_limit": 5e9
        >>>         },
        >>>     },
        >>>     "validation": {
        >>>         "strict": false
        >>>     },
        >>>     "verbose": true
        >>> }

    Example of a ~/.starfish.config file to disable caching:

        >>> {
        >>>     "slicedimage": {
        >>>         "caching": {
        >>>             "size_limit": 0
        >>>         }
        >>>     }
        >>> }

    """

    def __init__(self) -> None:
        """
        Loads the configuration specified by the STARFISH_CONFIG environment variable.

        Parameters
        ----------
        STARISH_CONFIG :
            This parameter is read from the environment to permit setting configuration
            values either directly or via a file. Keys read include:

             - ["slicedimage"]["caching"]["directory"]   (default: ~/.starfish/cache)
             - ["slicedimage"]["caching"]["size_limit"]  (default: None; 0 disables caching)
             - ["validation"]["strict"]                  (default: False)
             - ["verbose"]                               (default: True)

            Note: all keys can also be set by and environment variable constructed from the
            key parts and prefixed with STARFISH, e.g. STARFISH_VALIDATION_STRICT.
        """
        config = os.environ.get("STARFISH_CONFIG", "@~/.starfish/config")
        self._config_obj = Config(config)

        # If no directory is set, then force the default
        self._slicedimage = self._config_obj.lookup(("slicedimage",), NestedDict())
        if not self._slicedimage["caching"]["directory"]:
            self._slicedimage["caching"]["directory"] = "~/.starfish/cache"
        self._slicedimage_update(('caching', 'directory'))
        self._slicedimage_update(('caching', 'size_limit'), int)

        self._strict = self._config_obj.lookup(
            ("validation", "strict"), self.flag("STARFISH_VALIDATION_STRICT", "false"))

        self._verbose = self._config_obj.lookup(
            ("verbose",), self.flag("STARFISH_VERBOSE", "true"))

    def _slicedimage_update(self, lookup, parse=lambda x: x):
        """ accept STARFISH_SLICEDIMAGE_ or SLICEDIMAGE_ prefixes"""
        name = "SLICEDIMAGE_" + "_".join([x.upper() for x in lookup])
        if name not in os.environ:
            name = "STARFISH_" + name
            if name not in os.environ:
                return
        value = parse(os.environ[name])

        v = self._slicedimage
        for k in lookup[:-1]:
            v = v[k]
        v[lookup[-1]] = value

    @staticmethod
    def flag(name, default_value=""):
        value = os.environ.get(name, default_value)
        if isinstance(value, str):
            value = value.lower()
            return value in ("true", "1", "yes", "y", "on", "active", "enabled")

    @property
    def slicedimage(self):
        return dict(self._slicedimage)

    @property
    def strict(self):
        return self._strict

    @property
    def verbose(self):
        return self._verbose
