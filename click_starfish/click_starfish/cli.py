import click

from .registration._base import RegistrationBase


@click.group(name='starfish')
def cli():
    """create the outer CLI object"""
    pass


cli.add_command(RegistrationBase.registration)
