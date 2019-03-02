from collections import OrderedDict
from typing import Sequence, Tuple

import numpy as np
import xarray as xr
from scipy.ndimage.filters import gaussian_filter

from starfish import Codebook, ImageStack
from starfish.imagestack import physical_coordinate_calculator
from starfish.types import Axes, Coordinates, Features, PhysicalCoordinateTypes


def imagestack_with_coords_factory(stack_shape: OrderedDict, coords: OrderedDict) -> ImageStack:
    """
    Create an ImageStack of given shape and assigns the given x,y,z
    min/max physical coordinates to each tile.

    Parameters
    ----------
    stack_shape: OrderedDict
        Dict[Axes, int] defining the size of each dimension for an ImageStack

    coords: OrderedDict
        Dict[PhysicalCoordinateTypes, float] defining the min/max values of physical
        coordinates to assign to the Imagestack
    """

    stack = ImageStack.synthetic_stack(num_round=stack_shape[Axes.ROUND],
                                       num_ch=stack_shape[Axes.CH],
                                       num_z=stack_shape[Axes.ZPLANE],
                                       tile_height=stack_shape[Axes.Y],
                                       tile_width=stack_shape[Axes.X])

    stack.xarray[Coordinates.X.value] = xr.DataArray(
        np.linspace(coords[PhysicalCoordinateTypes.X_MIN], coords[PhysicalCoordinateTypes.X_MAX],
                    stack.xarray.sizes[Axes.X.value]), dims=Axes.X.value)

    stack.xarray[Coordinates.Y.value] = xr.DataArray(
        np.linspace(coords[PhysicalCoordinateTypes.Y_MIN], coords[PhysicalCoordinateTypes.Y_MAX],
                    stack.xarray.sizes[Axes.Y.value]), dims=Axes.Y.value)

    z_coord = physical_coordinate_calculator.\
        get_physical_coordinates_of_z_plane((coords[PhysicalCoordinateTypes.Z_MIN],
                                             coords[PhysicalCoordinateTypes.Z_MAX]))

    stack.xarray[Coordinates.Z.value] = xr.DataArray(np.zeros(
        stack.xarray.sizes[Axes.ZPLANE.value]),
        dims=Axes.ZPLANE.value)

    for z in stack.axis_labels(Axes.ZPLANE):
        stack.xarray[Coordinates.Z.value].loc[z] = z_coord

    return stack


def codebook_array_factory() -> Codebook:
    """
    Codebook with two codewords describing an experiment with three channels and two imaging rounds.
    Both codes have two "on" channels.
    """
    data = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 2, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    return Codebook.from_code_array(data)


def _create_dataset(
    pixel_dimensions: Tuple[int, int, int],
    spot_coordinates: Sequence[Tuple[int, int, int]],
    codebook: Codebook
) -> ImageStack:
    """
    creates a numpy array containing one spot per codebook entry at spot_coordinates. length of
    spot_coordinates must therefore match the number of codes in Codebook.
    """
    assert len(spot_coordinates) == codebook.sizes[Features.TARGET]

    data_shape = (
        codebook.sizes[Axes.ROUND.value],
        codebook.sizes[Axes.CH.value],
        *pixel_dimensions
    )
    imagestack_data = np.zeros((data_shape), dtype=np.float32)

    for ((z, y, x), f) in zip(spot_coordinates, range(codebook.sizes[Features.TARGET])):
        imagestack_data[:, :, z, y, x] = codebook[f].transpose(Axes.ROUND.value, Axes.CH.value)

    # blur with a small non-isotropic kernel TODO make kernel smaller.
    imagestack_data = gaussian_filter(imagestack_data, sigma=(0, 0, 0.7, 1.5, 1.5))
    return ImageStack.from_numpy_array(imagestack_data)


def two_spot_one_hot_coded_data_factory() -> Tuple[Codebook, ImageStack, float]:
    """
    Produce a 2-channel 2-round Codebook with two codes and an ImageStack containing one spot from
    each code. The spots do not overlap and the data are noiseless.

    The encoding of this data is similar to that used in In-situ Sequencing, FISSEQ,
    BaristaSeq, STARMAP, MExFISH, or SeqFISH.

    Returns
    -------
    Codebook :
        codebook containing codes that match the data
    ImageStack :
        noiseless ImageStack containing one spot per code in codebook
    float :
        the maximum intensity found in the created ImageStack

    """

    codebook_data = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    codebook = Codebook.from_code_array(codebook_data)

    imagestack = _create_dataset(
        pixel_dimensions=(10, 100, 100),
        spot_coordinates=((4, 10, 90), (5, 90, 10)),
        codebook=codebook
    )

    max_intensity = np.max(imagestack.xarray.values)

    return codebook, imagestack, max_intensity


def two_spot_sparse_coded_data_factory() -> Tuple[Codebook, ImageStack, float]:
    """
    Produce a 3-channel 3-round Codebook with two codes and an ImageStack containing one spot from
    each code. The spots do not overlap and the data are noiseless.

    These spots display sparsity in both rounds and channels, similar to the sparse encoding of
    MERFISH

    Returns
    -------
    ImageStack :
        noiseless ImageStack containing two spots

    """

    codebook_data = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 2, Axes.CH.value: 1, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 2, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    codebook = Codebook.from_code_array(codebook_data)

    imagestack = _create_dataset(
        pixel_dimensions=(10, 100, 100),
        spot_coordinates=((4, 10, 90), (5, 90, 10)),
        codebook=codebook
    )

    max_intensity = np.max(imagestack.xarray.values)

    return codebook, imagestack, max_intensity


def two_spot_informative_blank_coded_data_factory() -> Tuple[Codebook, ImageStack, float]:
    """
    Produce a 4-channel 2-round Codebook with two codes and an ImageStack containing one spot from
    each code. The spots do not overlap and the data are noiseless.

    The encoding of this data is essentially a one-hot encoding, but where one of the channels is a
    intentionally and meaningfully "blank".

    Returns
    -------
    Codebook :
        codebook containing codes that match the data
    ImageStack :
        noiseless ImageStack containing one spot per code in codebook
    float :
        the maximum intensity found in the created ImageStack

    """

    codebook_data = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1},
                # round 3 is blank and channel 3 is not used
            ],
            Features.TARGET: "GENE_A"
        },
        {
            Features.CODEWORD: [
                # round 0 is blank and channel 0 is not used
                {Axes.ROUND.value: 1, Axes.CH.value: 3, Features.CODE_VALUE: 1},
                {Axes.ROUND.value: 2, Axes.CH.value: 2, Features.CODE_VALUE: 1},
            ],
            Features.TARGET: "GENE_B"
        },
    ]
    codebook = Codebook.from_code_array(codebook_data)

    imagestack = _create_dataset(
        pixel_dimensions=(10, 100, 100),
        spot_coordinates=((4, 10, 90), (5, 90, 10)),
        codebook=codebook
    )

    max_intensity = np.max(imagestack.xarray.values)

    return codebook, imagestack, max_intensity
