from starfish.core.codebook.codebook import Codebook
from starfish.core.experiment.experiment import Experiment
from starfish.core.util.indirectfile._base import ConversionFormula


class GetCodebookFromExperiment(ConversionFormula):
    def applicable(self, input_parameter: str) -> bool:
        return input_parameter.startswith("@")

    def load(self, input_parameter: str) -> Codebook:
        path = input_parameter[1:]
        experiment = Experiment.from_json(path)
        return experiment.codebook


class GetCodebook(ConversionFormula):
    def applicable(self, input_parameter: str) -> bool:
        return not input_parameter.startswith("@")

    def load(self, input_parameter: str) -> Codebook:
        return Codebook.open_json(input_parameter)
