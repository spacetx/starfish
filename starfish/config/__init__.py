import os
from contextlib import contextmanager

from starfish.util.config import Config, NestedDict


@contextmanager
def environ(**kwargs):
    """Overrides environment variables (prefixed with "STARFISH_")
    for the duration of the call.

    Examples
    --------
    Turn on validation:

        >>> from starfish.config import environ
        >>> from starfish.experiment.experiment import Experiment
        >>> with environ(VALIDATION_STRICT="true"):
        >>>     Experiment.from_json(URL)

    """
    orig = dict()
    try:
        for k, newval in kwargs.items():
            if not k.startswith("STARFISH_"):
                k = "STARFISH_%s" % k
            old = os.environ.get(k, None)
            try:
                os.environ[k] = newval
                orig[k] = old  # Only store if successful
            except TypeError:
                raise
        yield
    finally:
        for k, oldval in orig.items():
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
    backend : dictionary
        Subdictionary that can be passed to the IO backend, e.g. slicedimage.
    strict : bool
        Whether or not loaded json should be validated.
    """

    def __init__(self) -> None:
        """
        Loads the configuration specified by the STARFISH_CONFIG environment variable.

        Parameters
        ----------
        STARISH_CONFIG :
            This parameter is read from the environment to permit setting configuration
            values either directly or via a file. Keys read include:

             - ["backend"]["caching"]["directory"]   (default: ~/.starfish-cache, enabling caching)
             - ["backend"]["caching"]["size_limit"]  (default: None)
             - ["validation"]["strict"]              (default: False)

            Note: all keys can also be set by and environment variable constructed from the
            key parts and prefixed with STARFISH, e.g. STARFISH_VALIDATION_STRICT.
        """
        self._config_obj = Config()  # STARFISH_CONFIG is assumed
        self._backend = self._config_obj.lookup(
            ("backend",), {'caching': {'directory': "~/.starfish-cache"}})
        self._backend_update(('caching', 'directory'))
        self._backend_update(('caching', 'size_limit'), int)

        self._strict = self._config_obj.lookup(
            ("validation", "strict"), self.flag("STARFISH_VALIDATION_STRICT"))

    def _backend_update(self, lookup, parse=lambda x: x):
        name = "STARFISH_BACKEND_" + "_".join([x.upper() for x in lookup])
        if name not in os.environ:
            return
        value = parse(os.environ[name])

        v = self._backend
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
    def backend(self):
        return dict(self._backend)

    @property
    def strict(self):
        return self._strict
