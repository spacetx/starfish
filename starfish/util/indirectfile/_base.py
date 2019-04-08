import abc
from typing import Generic, Iterable, TypeVar


RecipeResultType = TypeVar("RecipeResultType")


class ConversionRecipe(Generic[RecipeResultType]):
    @abc.abstractmethod
    def applicable(self, input_parameter: str) -> bool:
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


def convert(value: str, conversion_recipes: Iterable[ConversionRecipe]):
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
            f"Could not find applicable gonversion recipe for {value}"
        )
    raise NoSuccessfulConversionRecipeError(
        f"All applicable conversion recipes failed to run successfully for {value}."
    )
