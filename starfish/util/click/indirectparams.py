import abc
from typing import Generic, Iterable, TypeVar

from starfish.codebook.codebook import Codebook
from starfish.imagestack.imagestack import ImageStack
from starfish.util.indirectfile import (
    ConversionRecipe,
    convert,
    GetCodebook,
    GetCodebookFromExperiment,
    GetImageStack,
    GetImageStackFromExperiment,
    NoApplicableConversionRecipeError,
    NoSuccessfulConversionRecipeError,
)
from . import ParamType


IndirectResultType = TypeVar("IndirectResultType")


class IndirectFile(ParamType, Generic[IndirectResultType]):
    def convert(self, value: str, param, ctx):
        conversion_recipes = self.get_conversion_recipes()
        try:
            return convert(value, conversion_recipes)
        except (NoApplicableConversionRecipeError, NoSuccessfulConversionRecipeError) as ex:
            self.fail(ex.args[0])

    @abc.abstractmethod
    def get_conversion_recipes(self) -> Iterable[ConversionRecipe[IndirectResultType]]:
        """Return one or more conversion recipes to get from an input string to the type of object
        we want.
        """
        raise NotImplementedError()


class CodebookParam(IndirectFile[Codebook]):
    def __init__(self):
        self.name = "codebook"

    def get_conversion_recipes(self) -> Iterable[ConversionRecipe[Codebook]]:
        return [
            GetCodebookFromExperiment(),
            GetCodebook(),
        ]


CodebookParamType = CodebookParam()


class ImageStackParam(IndirectFile[ImageStack]):
    def __init__(self):
        self.name = "imagestack"

    def get_conversion_recipes(self) -> Iterable[ConversionRecipe[ImageStack]]:
        return [
            GetImageStackFromExperiment(),
            GetImageStack(),
        ]


ImageStackParamType = ImageStackParam()
