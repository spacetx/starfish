from starfish.core.imagestack.test.factories import synthetic_stack
from starfish.core.types import FunctionSource
from .. import Map


def test_map():
    """test that apply correctly applies a simple function across 2d tiles of a Stack"""
    stack = synthetic_stack()
    assert (stack.xarray == 1).all()
    mapper = Map("divide", 2)
    output = mapper.run(stack)
    assert (output.xarray == 0.5).all()

    mapper = Map(FunctionSource.np("divide"), 2)
    output = mapper.run(stack)
    assert (output.xarray == 0.5).all()
