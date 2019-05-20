import re

from starfish.core.experiment.experiment import Experiment
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.util.indirectfile._base import ConversionFormula


CRE = re.compile("@(?P<path>.+)\[(?P<fov>[^\[\]]+)\]\[(?P<image_type>[^\[\]]+)\]")  # noqa: W605


class GetImageStackFromExperiment(ConversionFormula[ImageStack]):
    def applicable(self, input_parameter: str) -> bool:
        return CRE.match(input_parameter) is not None

    def load(self, input_parameter: str) -> ImageStack:
        mo = CRE.match(input_parameter)
        assert mo is not None
        experiment = Experiment.from_json(mo.group("path"))
        fov = experiment[mo.group("fov")]
        return fov.get_image(mo.group("image_type"))


class GetImageStack(ConversionFormula[ImageStack]):
    def applicable(self, input_parameter: str) -> bool:
        return not CRE.match(input_parameter)

    def load(self, input_parameter: str) -> ImageStack:
        return ImageStack.from_path_or_url(input_parameter)
