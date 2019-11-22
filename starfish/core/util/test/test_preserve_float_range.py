import numpy as np
import pytest
import xarray as xr

from ..levels import preserve_float_range


def combos(make_xarrays: bool):
    test_parameters = (
        # none of these will change the data
        (
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
        ),

        # different dtype.
        (
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
        ),

        # some below zero stuff
        (
            np.asarray((-1.0, 0.0, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
        ),

        # some above one stuff
        (
            np.asarray((0.0, 1.0, 2.0), dtype=np.float32),
            np.asarray((0.0, 1.0, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.5, 1.0), dtype=np.float32),
        ),

        # both below zero and above one stuff
        (
            np.asarray((-1.0, 0.0, 1.0, 2.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 0.5, 1.0), dtype=np.float32),
        ),

        # both below zero and above one stuff, different dtype
        (
            np.asarray((-1.0, 0.0, 1.0, 2.0), dtype=np.float64),
            np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 0.5, 1.0), dtype=np.float32),
        ),

        # both below zero and above one stuff
        (
            np.asarray((-1, 0, 1, 2), dtype=np.int16),
            np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 0.5, 1.0), dtype=np.float32),
        ),
    )

    if make_xarrays:
        return (
            (xr.DataArray(source),
             xr.DataArray(expected_clipped_output),
             xr.DataArray(expected_rescaled_output))
            for source, expected_clipped_output, expected_rescaled_output in test_parameters
        )
    else:
        return test_parameters


@pytest.mark.parametrize(
    ("_source", "expected_clipped_output", "expected_rescaled_output"),
    combos(True)
)
def test_preserve_range_xarray(
        _source: xr.DataArray,
        expected_clipped_output: xr.DataArray,
        expected_rescaled_output: xr.DataArray,
):
    def run(source: xr.DataArray, expected_output: xr.DataArray, rescale: bool):
        original_data = source.copy(deep=True)

        # compute while trying to preserve the input.  if there is no change to the data, we should
        # come back with the same block of memory.
        output_preserve_input = preserve_float_range(source, rescale=rescale, preserve_input=True)
        assert expected_output.equals(output_preserve_input)
        assert original_data.equals(source)
        if source.dtype == np.float32 and expected_output.equals(source):
            assert output_preserve_input.values is source.values

        # compute it while trying to save memory.
        output_stingy_memory = preserve_float_range(source, rescale=rescale, preserve_input=False)
        assert expected_output.equals(output_stingy_memory)
        if source.dtype == np.float32:
            assert output_stingy_memory.values is source.values

    run(_source.copy(deep=True), expected_clipped_output, False)
    run(_source.copy(deep=True), expected_rescaled_output, True)


@pytest.mark.parametrize(
    ("_source", "expected_clipped_output", "expected_rescaled_output"),
    combos(False)
)
def test_preserve_range_ndarray(
        _source: np.ndarray,
        expected_clipped_output: np.ndarray,
        expected_rescaled_output: np.ndarray,
):
    def run(source: np.ndarray, expected_output: np.ndarray, rescale: bool):
        original_data = source.copy()

        # compute while trying to preserve the input.  if there is no change to the data, we should
        # come back with the same block of memory.
        output_preserve_input = preserve_float_range(source, rescale=rescale, preserve_input=True)
        assert np.all(np.equal(expected_output, output_preserve_input))
        assert np.all(np.equal(original_data, source))
        if source.dtype == np.float32 and np.all(np.equal(expected_output, source)):
            # no change
            assert output_preserve_input is source

        # compute it while trying to save memory.
        output_stingy_memory = preserve_float_range(source, rescale=rescale, preserve_input=False)
        assert np.all(np.equal(expected_output, output_stingy_memory))
        if source.dtype == np.float32:
            assert output_stingy_memory is source

    run(_source.copy(), expected_clipped_output, False)
    run(_source.copy(), expected_rescaled_output, True)
