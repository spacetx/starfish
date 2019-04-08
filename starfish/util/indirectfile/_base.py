import abc
from typing import Generic, Iterable, TypeVar


RecipeResultType = TypeVar("RecipeResultType")


class ConversionRecipe(Generic[RecipeResultType]):
    """A conversion recipe represents a plausible contract to convert a string value to another data
    type.  Each conversion recipe implements two methods -- a lightweight method to determine if
    this recipe can be applied given a string value, and a load method that actually does the
    conversion.
    """
    @abc.abstractmethod
    def applicable(self, input_parameter: str) -> bool:
        """Returns true iff this recipe might work for the given input."""
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, input_parameter: str) -> RecipeResultType:
        """Attempt to run this conversion recipe against this input."""
        raise NotImplementedError()


class NoApplicableConversionRecipeError(Exception):
    """Raised when no conversion recipe declared itself applicable to this input string."""
    pass


class NoSuccessfulConversionRecipeError(Exception):
    """Raised when all the conversion recipes that declared itself applicable failed to execute
    successfully."""
    pass


ConvertResultType = TypeVar("ConvertResultType")


def convert(
        value: str,
        conversion_recipes: Iterable[ConversionRecipe[ConvertResultType]]) -> ConvertResultType:
    """
    Given a string value and a series of conversion recipes, attempt to convert the value using the
    recipes.

    If none of the recipes declare themselves as applicable, then raise
    :py:class:`NoApplicableConversionRecipeError`.  If none of the recipes that declare themselves
    as eligible run successfully, then raise :py:class:`NoSuccessfulConversionRecipeError`.

    Parameters
    ----------
    value : str
        The string value we are attempting to convert.
    conversion_recipes : Iterable[ConversionRecipe[ConvertResultType]]
        A series of conversion recipes.

    Returns
    -------
    The converted value.
    """
    none_applied = True

    for conversion_recipe in conversion_recipes:
        if conversion_recipe.applicable(value):
            none_applied = False
            try:
                return conversion_recipe.load(value)
            except Exception:
                pass

    if none_applied:
        raise NoApplicableConversionRecipeError(
            f"Could not find applicable gonversion recipe for {value}")
    raise NoSuccessfulConversionRecipeError(
        f"All applicable conversion recipes failed to run successfully for {value}.")
