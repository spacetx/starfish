import click

from .._pipeline_component import PipelineComponent


class RegistrationBase(PipelineComponent):

    @staticmethod
    @click.group()
    def registration():
        """
        Create registration category in the base class of the PipelineComponent.
        This is equivalent to __init__.py in the API.
        """
        pass  # can't be NotImplementedError; could also go into outer scope for consistency.
