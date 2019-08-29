import numpy as np
import xarray as xr

from starfish import DecodedIntensityTable, IntensityTable
from starfish.core.codebook.test.factories import codebook_array_factory, loaded_codebook
from starfish.core.types import Coordinates, Features
from ..overlap import Area


def synthetic_intensity_table() -> IntensityTable:
    return IntensityTable.synthetic_intensities(loaded_codebook(), n_spots=2)


def synthetic_decoded_intensity_table(
        codebook,
        num_z: int = 12,
        height: int = 50,
        width: int = 40,
        n_spots: int = 10,
        mean_fluor_per_spot: int = 200,
        mean_photons_per_fluor: int = 50,
) -> DecodedIntensityTable:
    """
    Creates an IntensityTable with synthetic spots, that correspond to valid
    codes in a provided codebook.

    Parameters
    ----------
    codebook : Codebook
        Starfish codebook object.
    num_z : int
        Number of z-planes to use when localizing spots.
    height : int
        y dimension of each synthetic plane.
    width : int
        x dimension of each synthetic plane.
    n_spots : int
        Number of spots to generate.
    mean_fluor_per_spot : int
         Mean number of fluorophores per spot.
    mean_photons_per_fluor : int
        Mean number of photons per fluorophore.

    Returns
    -------
    DecodedIntensityTable
    """

    intensities = IntensityTable.synthetic_intensities(
        codebook,
        num_z=num_z,
        height=height,
        width=width,
        n_spots=n_spots,
        mean_fluor_per_spot=mean_fluor_per_spot,
        mean_photons_per_fluor=mean_photons_per_fluor
    )
    targets = np.random.choice(
        codebook.coords[Features.TARGET], size=n_spots, replace=True)

    return DecodedIntensityTable.from_intensity_table(intensities, targets=(Features.AXIS, targets))


def assign_synthetic_targets(intensities: IntensityTable) -> DecodedIntensityTable:
    """
    Assign fake target values to the given IntensityTable

    Parameters
    ----------
    intensities : IntensityTable
        intensity_table to assign target values to

    Returns
    -------
    DecodedIntensityTable
    """
    intensities = DecodedIntensityTable(intensities)
    return DecodedIntensityTable.from_intensity_table(
        intensities,
        targets=(Features.AXIS, np.random.choice(list('ABCD'), size=20)),
        distances=(Features.AXIS, np.random.rand(20)))


def create_intensity_table_with_coords(area: Area, n_spots: int=10) -> IntensityTable:
    """
    Creates a 50X50 intensity table with physical coordinates within
    the given Area.

    Parameters
    ----------
    area: Area
        The area of physical space the IntensityTable should be defined over
    n_spots:
        Number of spots to add to the IntensityTable
    """
    codebook = codebook_array_factory()
    it = IntensityTable.synthetic_intensities(
        codebook,
        num_z=1,
        height=50,
        width=50,
        n_spots=n_spots
    )
    # intensity table 1 has 10 spots, xmin = 0, ymin = 0, xmax = 2, ymax = 1
    it[Coordinates.X.value] = xr.DataArray(np.linspace(area.min_x, area.max_x, n_spots),
                                           dims=Features.AXIS)
    it[Coordinates.Y.value] = xr.DataArray(np.linspace(area.min_y, area.max_y, n_spots),
                                           dims=Features.AXIS)
    return it
