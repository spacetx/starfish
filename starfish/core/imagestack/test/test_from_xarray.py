import numpy as np
import pytest
import xarray as xr

from starfish import ImageStack
from starfish.types import Axes, Coordinates


def test_from_xarray_basic():
    """Test basic from_xarray functionality."""
    # Create an ImageStack using from_numpy
    array = np.random.rand(2, 3, 4, 50, 60).astype(np.float32)
    stack1 = ImageStack.from_numpy(array)

    # Get its xarray
    xarr = stack1.xarray

    # Create a new ImageStack from the xarray
    stack2 = ImageStack.from_xarray(xarr)

    # Verify they have the same data
    assert np.array_equal(stack1.xarray.values, stack2.xarray.values)
    assert stack1.xarray.shape == stack2.xarray.shape
    assert stack1.xarray.dims == stack2.xarray.dims


def test_from_xarray_preserves_coordinates():
    """Test that from_xarray preserves physical coordinates."""
    # Create an ImageStack with specific coordinates
    array = np.random.rand(2, 3, 4, 50, 60).astype(np.float32)
    coordinates = {
        Coordinates.X: np.linspace(0, 15, 60),
        Coordinates.Y: np.linspace(0, 20, 50),
        Coordinates.Z: [1, 2, 15, 20],
    }
    stack1 = ImageStack.from_numpy(array, coordinates=coordinates)

    # Reconstruct from xarray
    stack2 = ImageStack.from_xarray(stack1.xarray)

    # Verify coordinates are preserved
    assert np.array_equal(
        stack1.xarray.coords[Coordinates.X.value].values,
        stack2.xarray.coords[Coordinates.X.value].values
    )
    assert np.array_equal(
        stack1.xarray.coords[Coordinates.Y.value].values,
        stack2.xarray.coords[Coordinates.Y.value].values
    )
    assert np.array_equal(
        stack1.xarray.coords[Coordinates.Z.value].values,
        stack2.xarray.coords[Coordinates.Z.value].values
    )


def test_from_xarray_preserves_index_labels():
    """Test that from_xarray preserves custom index labels."""
    # Create an ImageStack with custom index labels
    array = np.random.rand(2, 3, 4, 50, 60).astype(np.float32)
    index_labels = {
        Axes.ROUND: [10, 20],
        Axes.CH: [0, 2, 5],
        Axes.ZPLANE: [1, 3, 7, 9],
    }
    stack1 = ImageStack.from_numpy(array, index_labels=index_labels)

    # Reconstruct from xarray
    stack2 = ImageStack.from_xarray(stack1.xarray)

    # Verify index labels are preserved
    assert list(stack2.xarray.coords[Axes.ROUND.value].values) == index_labels[Axes.ROUND]
    assert list(stack2.xarray.coords[Axes.CH.value].values) == index_labels[Axes.CH]
    assert list(stack2.xarray.coords[Axes.ZPLANE.value].values) == index_labels[Axes.ZPLANE]


def test_from_xarray_concatenation():
    """Test using from_xarray to concatenate ImageStacks."""
    # Create two ImageStacks
    array1 = np.random.rand(2, 3, 4, 50, 60).astype(np.float32)
    array2 = np.random.rand(3, 3, 4, 50, 60).astype(np.float32)

    index_labels1 = {
        Axes.ROUND: [0, 1],
        Axes.CH: [0, 1, 2],
        Axes.ZPLANE: [0, 1, 2, 3],
    }
    index_labels2 = {
        Axes.ROUND: [2, 3, 4],
        Axes.CH: [0, 1, 2],
        Axes.ZPLANE: [0, 1, 2, 3],
    }

    stack1 = ImageStack.from_numpy(array1, index_labels=index_labels1)
    stack2 = ImageStack.from_numpy(array2, index_labels=index_labels2)

    # Concatenate along round dimension
    concatenated_xarr = xr.concat([stack1.xarray, stack2.xarray], dim=Axes.ROUND.value)

    # Create new ImageStack from concatenated xarray
    combined_stack = ImageStack.from_xarray(concatenated_xarr)

    # Verify the combined stack has the right shape
    assert combined_stack.xarray.sizes[Axes.ROUND.value] == 5
    assert combined_stack.xarray.sizes[Axes.CH.value] == 3
    assert combined_stack.xarray.sizes[Axes.ZPLANE.value] == 4
    assert combined_stack.xarray.sizes[Axes.Y.value] == 50
    assert combined_stack.xarray.sizes[Axes.X.value] == 60

    # Verify round labels are correct
    assert list(combined_stack.xarray.coords[Axes.ROUND.value].values) == [0, 1, 2, 3, 4]


def test_from_xarray_raises_error_on_missing_dimensions():
    """Test that from_xarray raises an error when dimensions are missing."""
    # Create a DataArray with missing dimension
    data = xr.DataArray(
        np.random.rand(2, 3, 50, 60).astype(np.float32),
        dims=['r', 'c', 'y', 'x'],  # Missing 'z'
        coords={'r': [0, 1], 'c': [0, 1, 2]}
    )

    with pytest.raises(ValueError, match="must have dimensions"):
        ImageStack.from_xarray(data)


def test_from_xarray_raises_error_on_missing_coordinates():
    """Test that from_xarray raises an error when required coordinates are missing."""
    # Create a DataArray without coordinate labels
    data = xr.DataArray(
        np.random.rand(2, 3, 4, 50, 60).astype(np.float32),
        dims=['r', 'c', 'z', 'y', 'x'],
        # Missing coordinate definitions for r, c, z
    )

    with pytest.raises(ValueError, match="must have coordinates"):
        ImageStack.from_xarray(data)


def test_from_xarray_handles_type_conversion():
    """Test that from_xarray converts non-float32 data."""
    # Create an ImageStack with uint16 data
    array = np.random.randint(0, 1000, (2, 3, 4, 50, 60), dtype=np.uint16)
    stack1 = ImageStack.from_numpy(array)

    # This should have been converted to float32
    assert stack1.xarray.dtype == np.float32

    # Now create from the xarray (which is already float32)
    stack2 = ImageStack.from_xarray(stack1.xarray)

    assert stack2.xarray.dtype == np.float32


def test_from_xarray_round_trip():
    """Test that we can do multiple round trips between numpy, xarray, and ImageStack."""
    # Start with numpy
    array = np.random.rand(2, 3, 4, 50, 60).astype(np.float32)

    # numpy -> ImageStack
    stack1 = ImageStack.from_numpy(array)

    # ImageStack -> xarray
    xarr1 = stack1.xarray

    # xarray -> ImageStack
    stack2 = ImageStack.from_xarray(xarr1)

    # ImageStack -> xarray
    xarr2 = stack2.xarray

    # xarray -> ImageStack
    stack3 = ImageStack.from_xarray(xarr2)

    # Verify all stacks have the same data
    assert np.array_equal(stack1.xarray.values, stack2.xarray.values)
    assert np.array_equal(stack2.xarray.values, stack3.xarray.values)
