import os

from starfish.util.config import Config


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

        self._strict = self._config_obj.lookup(
            ("validation", "strict"), self.flag("STARFISH_VALIDATION_STRICT"))

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
