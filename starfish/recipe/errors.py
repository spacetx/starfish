class RecipeWarning(RuntimeWarning):
    pass


class RecipeError(Exception):
    pass


class ConstructorExtraParameterWarning(RecipeWarning):
    """Raised when a recipe contains parameters that an algorithms constructor does not expect."""


class TypeInferenceError(RecipeError):
    """Raised when we cannot infer the type of object an algorithm expects in its constructor or
    its run method."""


class ConstructorError(RecipeError):
    """Raised when there is an error raised during the construction of an algorithm class."""
    pass


class RunInsufficientParametersError(RecipeError):
    """Raised when the recipe does not provide sufficient parameters for the run method."""


class ExecutionError(RecipeError):
    """Raised when there is an error raised during the execution of an algorithm."""
    pass
