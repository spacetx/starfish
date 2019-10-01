import warnings
from itertools import product
from typing import Tuple

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import img_as_float32, img_as_uint

from starfish import Codebook, ImageStack, IntensityTable
from starfish.core.image.Filter.white_tophat import WhiteTophat
from starfish.core.imagestack.test.factories import create_imagestack_from_codebook
from starfish.core.spots.DecodeSpots import MetricDistance
from starfish.core.spots.FindSpots import BlobDetector
from starfish.core.types import Axes, Features


class SyntheticData:
    """Synthetic generator for spot data and container objects

    Currently, this generator only generates codebooks with single channels "on" in each
    imaging round. Therefore, it can generate ISS and smFISH type experiments, but not
    more complex codes.

    Examples
    --------
    >>> from starfish.core.test.factories import SyntheticData

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
    <starfish.imagestack.imagestack.ImageStack at 0x10a60c5f8>

    """

    def __init__(
            self,
            n_round: int = 4,
            n_ch: int = 4,
            n_z: int = 10,
            height: int = 50,
            width: int = 45,
            n_codes: int = 16,
            n_spots: int = 20,
            n_photons_background: int = 1000,
            background_electrons: int = 1,
            point_spread_function: Tuple[int, ...] = (4, 2, 2),
            camera_detection_efficiency: float = 0.25,
            gray_level: float = 37000.0 / 2 ** 16,
            ad_conversion_bits: int = 16,
            mean_fluor_per_spot: int = 200,
            mean_photons_per_fluor: int = 50,

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
        intensities = IntensityTable.synthetic_intensities(
            codebook, self.n_z, self.height, self.width, self.n_spots,
            self.mean_fluor_per_spot, self.mean_photons_per_fluor)
        assert intensities.dtype == np.float32 and intensities.max() <= 1
        return intensities

    def spots(self, intensities=None) -> ImageStack:
        if intensities is None:
            intensities = self.intensities()
        stack = SyntheticData.synthetic_spots(
            intensities, self.n_z, self.height, self.width, self.n_photons_background,
            self.point_spread_function, self.camera_detection_efficiency,
            self.background_electrons, self.gray_level, self.ad_coversion_bits)
        assert stack.xarray.dtype == np.float32
        return stack

    @classmethod
    def synthetic_spots(
            cls, intensities: IntensityTable, num_z: int, height: int, width: int,
            n_photons_background=1000, point_spread_function=(4, 2, 2),
            camera_detection_efficiency=0.25, background_electrons=1,
            graylevel: float = 37000.0 / 2 ** 16, ad_conversion_bits=16,
    ) -> ImageStack:
        """Generate a synthetic ImageStack from a set of Features stored in an IntensityTable

        Parameters
        ----------
        intensities : IntensityTable
            IntensityTable containing coordinates of fluorophores. Used to position and generate
            spots in the output ImageStack
        num_z : int
            Number of z-planes in the ImageStack
        height : int
            Height in pixels of the ImageStack
        width : int
            Width in pixels of the ImageStack
        n_photons_background : int
            Poisson rate for the number of background photons to add to each pixel of the image.
            Set this parameter to 0 to eliminate background.
            (default 1000)
        point_spread_function : Tuple[int]
            The width of the gaussian density wherein photons spread around their light source.
            Set to zero to eliminate this (default (4, 2, 2))
        camera_detection_efficiency : float
            The efficiency of the camera to detect light. Set to 1 to remove this filter (default
            0.25)
        background_electrons : int
            Poisson rate for the number of spurious electrons detected per pixel during image
            capture by the camera (default 1)
        graylevel : float
            The number of shades of gray displayable by the synthetic camera. Larger numbers will
            produce higher resolution images (default 37000 / 2 ** 16)
        ad_conversion_bits : int
            The number of bits used during analog to visual conversion (default 16)

        Returns
        -------
        ImageStack :
            synthetic spots

        """
        # check some params
        if not 0 < camera_detection_efficiency <= 1:
            raise ValueError(
                f'invalid camera_detection_efficiency value: {camera_detection_efficiency}. '
                f'Must be in the interval (0, 1].')

        def select_uint_dtype(array):
            """choose appropriate dtype based on values of an array"""
            max_val = np.max(array)
            for dtype in (np.uint8, np.uint16, np.uint32):
                if max_val <= np.iinfo(dtype).max:
                    return array.astype(dtype)
            raise ValueError('value exceeds dynamic range of largest skimage-supported type')

        # make sure requested dimensions are large enough to support intensity values
        axis_to_size = zip((Axes.ZPLANE.value, Axes.Y.value, Axes.X.value), (num_z, height, width))
        for axis, requested_size in axis_to_size:
            required_size = intensities.coords[axis].values.max() + 1
            if required_size > requested_size:
                raise ValueError(
                    f'locations of intensities contained in table exceed the size of requested '
                    f'axis {axis}. Required size {required_size} > {requested_size}.')

        # create an empty array of the correct size
        image = np.zeros(
            (
                intensities.sizes[Axes.ROUND.value],
                intensities.sizes[Axes.CH.value],
                num_z,
                height,
                width
            ), dtype=np.uint32
        )

        # starfish uses float images, but the logic here requires uint. We cast, and will cast back
        # at the end of the function
        intensities.values = img_as_uint(intensities)

        for round_, ch in product(*(range(s) for s in intensities.shape[1:])):
            spots = intensities[{Axes.ROUND.value: round_, Axes.CH.value: ch}]

            # numpy deprecated casting a specific way of casting floats that is triggered in xarray
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                values = spots.where(spots, drop=True)

            image[round_, ch, values.z, values.y.astype(int), values.x.astype(int)] = values

        intensities.values = img_as_float32(intensities)

        # add imaging noise
        image += np.random.poisson(n_photons_background, size=image.shape).astype(np.uint32)

        # blur image over coordinates, but not over round_/channels (dim 0, 1)
        sigma = (0, 0) + point_spread_function
        image = gaussian_filter(image, sigma=sigma, mode='nearest')

        image = image * camera_detection_efficiency

        image += np.random.normal(scale=background_electrons, size=image.shape)

        # mimic analog to digital conversion
        image = (image / graylevel).astype(int).clip(0, 2 ** ad_conversion_bits)

        # clip in case we've picked up some negative values
        image = np.clip(image, 0, a_max=None)

        # set the smallest int datatype that supports the data's intensity range
        image = select_uint_dtype(image)

        # convert to float for ImageStack
        with warnings.catch_warnings():
            # possible precision loss when casting from uint to float is acceptable
            warnings.simplefilter('ignore', UserWarning)
            image = img_as_float32(image)

        return ImageStack.from_numpy(image)


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

    imagestack = create_imagestack_from_codebook(
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

    imagestack = create_imagestack_from_codebook(
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

    imagestack = create_imagestack_from_codebook(
        pixel_dimensions=(10, 100, 100),
        spot_coordinates=((4, 10, 90), (5, 90, 10)),
        codebook=codebook
    )

    max_intensity = np.max(imagestack.xarray.values)

    return codebook, imagestack, max_intensity


def synthetic_dataset_with_truth_values():
    np.random.seed(2)
    synthesizer = SyntheticData(n_spots=5)
    codebook = synthesizer.codebook()
    true_intensities = synthesizer.intensities(codebook=codebook)
    image = synthesizer.spots(intensities=true_intensities)

    return codebook, true_intensities, image


def synthetic_dataset_with_truth_values_and_called_spots():
    codebook, true_intensities, image = synthetic_dataset_with_truth_values()

    wth = WhiteTophat(masking_radius=15)
    filtered = wth.run(image, in_place=False)

    min_sigma = 1.5
    max_sigma = 4
    num_sigma = 10
    threshold = 1e-4
    gsd = BlobDetector(min_sigma=min_sigma,
                       max_sigma=max_sigma,
                       num_sigma=num_sigma,
                       threshold=threshold,
                       measurement_type='max')
    spots = gsd.run(image_stack=filtered)
    decoder = MetricDistance(codebook=codebook, max_distance=1, min_intensity=0, norm_order=2)
    decoded_intensities = decoder.run(spots)
    assert decoded_intensities.shape[0] == 5
    return codebook, true_intensities, image, decoded_intensities


def synthetic_spot_pass_through_stack():
    codebook, true_intensities, _ = synthetic_dataset_with_truth_values()
    true_intensities = true_intensities[:2]
    # transfer the intensities to the stack but don't do anything to them.
    img_stack = SyntheticData.synthetic_spots(
        true_intensities, num_z=12, height=50, width=45, n_photons_background=0,
        point_spread_function=(0, 0, 0), camera_detection_efficiency=1.0,
        background_electrons=0, graylevel=1)
    return codebook, true_intensities, img_stack


def codebook_intensities_image_for_single_synthetic_spot():
    sd = SyntheticData(
        n_round=2, n_ch=2, n_z=2, height=20, width=30, n_codes=1, n_spots=1
    )
    codebook = sd.codebook()
    intensities = sd.intensities(codebook)
    image = sd.spots(intensities)
    return codebook, intensities, image
