import click

from ._base import RegistrationBase
from .._util import Singleton

import numpy as np  # not used; importing this to demonstrate exclusion from global import namespace


class _FourierTransform(RegistrationBase, metaclass=Singleton):
    """Implementing algorithm of the Registration pipeline component"""

    def __call__(self, input_json, output_json, *args, **kwargs) -> None:
        print(f'input={input_json}, output={output_json}')


# new required boilerplate starts here

# this object gets imported, it's a singleton of our class which is callable
fourier_transform = _FourierTransform()


# this object is registered to the CLI
@RegistrationBase.registration.command(name="fourier-transform")
@click.option("--input-json", help="input json")
@click.option("--output-json", help="output file")
def _cli(input_json, output_json):
    fourier_transform(input_json, output_json)
