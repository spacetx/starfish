from starfish import IntensityTable
from starfish.codebook.test.factories import loaded_codebook


def synthetic_intensity_table() -> IntensityTable:
    return IntensityTable.synthetic_intensities(loaded_codebook(), n_spots=2)
