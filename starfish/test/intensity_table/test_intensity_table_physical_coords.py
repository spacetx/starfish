
import xarray as xr
import numpy as np

from starfish import Codebook, IntensityTable, ImageStack
from starfish.imagestack import physical_coordinate_calculator
from starfish.types import Features, Indices, PHYSICAL_COORDINATE_DIMENSION, PhysicalCoordinateTypes


def imagestack_with_coords_factory() -> ImageStack:
    coords_array = xr.DataArray(
        np.empty(
            shape=(3, 2, 1, 6),
            dtype=np.float32,
        ),
        dims=(Indices.ROUND.value,
              Indices.CH.value,
              Indices.Z.value,
              PHYSICAL_COORDINATE_DIMENSION),
        coords={
            PHYSICAL_COORDINATE_DIMENSION: [
                PhysicalCoordinateTypes.X_MIN.value,
                PhysicalCoordinateTypes.X_MAX.value,
                PhysicalCoordinateTypes.Y_MIN.value,
                PhysicalCoordinateTypes.Y_MAX.value,
                PhysicalCoordinateTypes.Z_MIN.value,
                PhysicalCoordinateTypes.Z_MAX.value,
            ],
        },
    )

    coords_array.loc[0, 0, 0] = np.array([1, 2, 4, 6, 1, 3])

    stack = ImageStack.synthetic_stack(3, 2, 1, 50, 40)

    stack._coordinates = coords_array

    return stack


def codebook_array_factory() -> Codebook:
    """
    Codebook with two codewords describing an experiment with three channels and two imaging rounds.
    Both codes have two "on" channels.
    """
    data = [
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 0, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 2, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    return Codebook.from_code_array(data)


def test_tranfering_physical_coords_to_intensity_table():
    stack = imagestack_with_coords_factory()
    codebook = codebook_array_factory()

    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=1,
        height=50,
        width=40,
        n_spots=2
    )

    intensities = physical_coordinate_calculator.\
        transfer_physical_coords_from_imagestack_to_intensity_table(stack, intensities)
