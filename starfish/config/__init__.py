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
        STARFISH_STRICT_LOADING :
             This parameter is read from the environment. If set, then all JSON loaded by this
             method will be passed to the appropriate validator. The `strict` parameter to this
             method has priority over the environment variable.
        """
        self._config_obj = Config()  # STARFISH_CONFIG is assumed
        self._backend = self._config_obj.lookup(
            ("backend",), {'caching': {'directory': "~/.starfish-cache"}})

        self._strict = self._config_obj.lookup(
            ("validation", "strict"), os.environ.get("STARFISH_STRICT_LOADING", None))

    @property
    def backend(self):
        return dict(self._backend)

    # TODO: remove all strict arguments in favor of a property here
    def strict(self, strict=False):
        if self._strict is None:
            return strict
        return self._strict
