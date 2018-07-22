from typing import Tuple

from starfish.codebook import Codebook
from starfish.intensity_table import IntensityTable
from starfish.image import ImageStack


class SyntheticData:
    """Synthetic generator for spot data and container objects

    Currently, this generator only generates codebooks with single channels "on" in each
    imaging round. Therefore, it can generate ISS and smFISH type experiments, but not
    more complex codes.

    Examples
    --------
    >>> from starfish.util.synthesize import SyntheticData

    >>> sd = SyntheticData(n_ch=3, n_round=4, n_codes=2)
    >>> codes = sd.codebook()
    >>> codes
    <xarray.Codebook (target: 2, c: 3, h: 4)>
    array([[[0, 0, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0]],

           [[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 1]]], dtype=uint8)
    Coordinates:
      * target     (target) object 08b1a822-a1b4-4e06-81ea-8a4bd2b004a9 ...
      * c          (c) int64 0 1 2
      * h          (h) int64 0 1 2 3

    >>> # note that the features are drawn from the codebook
    >>> intensities = sd.intensities(codebook=codes)
    >>> intensities
    <xarray.IntensityTable (features: 2, c: 3, h: 4)>
    array([[[    0.,     0.,     0.,     0.],
            [    0.,     0.,  8022., 12412.],
            [11160.,  9546.,     0.,     0.]],

           [[    0.,     0.,     0.,     0.],
            [    0.,     0., 10506., 10830.],
            [11172., 12331.,     0.,     0.]]])
    Coordinates:
      * features   (features) MultiIndex
      - z          (features) int64 7 3
      - y          (features) int64 14 32
      - x          (features) int64 32 15
      - r          (features) float64 nan nan
      * c          (c) int64 0 1 2
      * h          (h) int64 0 1 2 3
        target     (features) object 08b1a822-a1b4-4e06-81ea-8a4bd2b004a9 ...

    >>> sd.spots(intensities=intensities)
    <starfish.image._stack.ImageStack at 0x10a60c5f8>

    """

    def __init__(
            self,
            n_round: int=4,
            n_ch: int=4,
            n_z: int=10,
            height: int=50,
            width: int=45,
            n_codes: int=16,
            n_spots: int=20,
            n_photons_background: int=1000,
            background_electrons: int=1,
            point_spread_function: Tuple[int, ...]=(4, 2, 2),
            camera_detection_efficiency: float=0.25,
            gray_level: float=37000.0 / 2 ** 16,
            ad_conversion_bits: int=16,
            mean_fluor_per_spot: int=200,
            mean_photons_per_fluor: int=50,

    ) -> None:
        self.n_round = n_round
        self.n_ch = n_ch
        self.n_z = n_z
        self.height = height
        self.width = width
        self.n_codes = n_codes
        self.n_spots = n_spots
        self.n_photons_background = n_photons_background
        self.background_electrons = background_electrons
        self.point_spread_function = point_spread_function
        self.camera_detection_efficiency = camera_detection_efficiency
        self.gray_level = gray_level
        self.ad_coversion_bits = ad_conversion_bits
        self.mean_fluor_per_spot = mean_fluor_per_spot
        self.mean_photons_per_fluor = mean_photons_per_fluor

    def codebook(self) -> Codebook:
        return Codebook.synthetic_one_hot_codebook(self.n_round, self.n_ch, self.n_codes)

    def intensities(self, codebook=None) -> IntensityTable:
        if codebook is None:
            codebook = self.codebook()
        return IntensityTable.synthetic_intensities(
            codebook, self.n_z, self.height, self.width, self.n_spots,
            self.mean_fluor_per_spot, self.mean_photons_per_fluor)

    def spots(self, intensities=None) -> ImageStack:
        if intensities is None:
            intensities = self.intensities()
        return ImageStack.synthetic_spots(
            intensities, self.n_z, self.height, self.width, self.n_photons_background,
            self.point_spread_function, self.camera_detection_efficiency,
            self.background_electrons, self.gray_level, self.ad_coversion_bits)
