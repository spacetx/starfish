from typing import Callable, Tuple

import numpy as np
import xarray as xr
import pandas as pd
from skimage.exposure import rescale_intensity
from scipy.ndimage.filters import gaussian_filter

from starfish.constants import Indices, Coordinates
from starfish.pipeline.features.codebook import Codebook


# defaults for stack size
DEFAULT_NUM_HYB = 4
DEFAULT_NUM_CH = 4
DEFAULT_NUM_Z = 10
DEFAULT_HEIGHT = 50
DEFAULT_WIDTH = 45

# defaults for codebook
N_CODES = 16

# constants for spot generation
N_SPOTS = 5                         # Number of spots to generate
N_PHOTONS_BACKGROUND = 1000         # Number of photons per pixel of background noise
BACKGROUND_ELECTRONS = 1            # camera read noise per pixel in units electrons
POINT_SPREAD_FUNCTION = (4, 2, 2)   # standard devitation of gaussian point spread function in pixel units
CAMERA_DETECTION_EFFICIENCY = 0.25  # quantum efficiency of the camera detector units number of electrons per photon
GRAYLEVEL = 37000.0 / 2 ** 16       # dynamic range of camera sensor 37,000 assuming a 16-bit AD converter
AD_CONVERSION_BITS = 16             # bit-size of analog to digital converter
MEAN_FLUOR_PER_SPOT = 200           # mean flours per transcripts - depends on amplification strategy (e.g HCR, bDNA)
MEAN_PHOTONS_PER_FLUOR = 50         # mean photons per flourophore - depends on exposure time, bleaching rate of dye


def select_uint_dtype(array):
    """choose appropriate dtype based on values of an array"""
    max_val = np.max(array)
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if max_val <= dtype(-1):
            return array.astype(dtype)
    raise ValueError('value exceeds dynamic range of largest numpy type')


# TODO not sure what to do with this xarray thing
def blank_imagestack(
        n_hyb: int=DEFAULT_NUM_HYB,
        n_ch: int=DEFAULT_NUM_CH,
        num_z: int=DEFAULT_NUM_Z,
        height: int=DEFAULT_HEIGHT,
        width: int=DEFAULT_WIDTH,
):
    data = np.zeros((n_hyb, n_ch, num_z, height, width))
    coords = [np.arange(s) for s in data.shape]
    dims = (Indices.HYB.value, Indices.CH.value, Indices.Z.value, Coordinates.Y.value, Coordinates.X.value)
    return xr.DataArray(data=data, coords=coords, dims=dims)


# TODO these are synthetic SpotAttributes
def create_spots(
        codebook: Codebook,
        n_hyb: int=DEFAULT_NUM_HYB,
        n_ch: int=DEFAULT_NUM_CH,
        num_z: int=DEFAULT_NUM_Z,
        height: int=DEFAULT_HEIGHT,
        width: int=DEFAULT_WIDTH,
        n_spots=N_SPOTS,
        mean_fluor_per_spot=MEAN_FLUOR_PER_SPOT,
        mean_photons_per_fluor=MEAN_PHOTONS_PER_FLUOR):

    image = blank_imagestack(n_hyb, n_ch, num_z, height, width)
    genes = np.random.choice(codebook.coords['gene_name'], size=n_spots, replace=True)
    _, hyb_indices, ch_indices = np.where(codebook[genes])

    # right now there is no jitter on x-y positions of the spots, we might want to make it a vector
    z = np.random.randint(0, num_z, size=n_spots)
    y = np.random.randint(0, height, size=n_spots)
    x = np.random.randint(0, width, size=n_spots)

    intensities = np.ones(shape=(len(hyb_indices), len(ch_indices), len(z), len(y), len(x)))
    intensities *= np.random.poisson(mean_photons_per_fluor, size=intensities.shape)
    intensities *= np.random.poisson(mean_fluor_per_spot, size=intensities.shape)

    image[hyb_indices, ch_indices, z, y, x] = intensities

    spot_attributes = pd.DataFrame({'gene': genes, 'z': z, 'y': y, 'x': x})

    return image, spot_attributes


# todo this is a synthetic ImageStack
def _synthetic_spots(
        n_hyb: int=DEFAULT_NUM_HYB,
        n_ch: int=DEFAULT_NUM_CH,
        num_z: int=DEFAULT_NUM_Z,
        height: int=DEFAULT_HEIGHT,
        width: int=DEFAULT_WIDTH,
        code_generator: Callable=Codebook.synthetic_one_hot_codes,
        n_photons_background=N_PHOTONS_BACKGROUND,
        point_spread_function=POINT_SPREAD_FUNCTION,
        camera_detection_efficiency=CAMERA_DETECTION_EFFICIENCY,
        background_electrons=BACKGROUND_ELECTRONS,
        graylevel=GRAYLEVEL,
        ad_conversion_bits=AD_CONVERSION_BITS
) -> Tuple[xr.DataArray, pd.DataFrame]:
    """

    Parameters
    ----------
    code_generator
    n_photons_background
    point_spread_function
    camera_detection_efficiency
    background_electrons
    graylevel
    ad_conversion_bits

    Returns
    -------
    np.ndarray[uint] :
        5-d tensor with uint kind data. The type of uint depends on the maximum intensity of the array, the dtype will
        be shrunk so that this funciton generates the smallest dtype that contains the data without clipping or
        wrapping.

    """
    codebook = code_generator(n_hyb, n_ch, n_codes=12)

    image, spot_attributes = create_spots(codebook, n_hyb, n_ch, num_z, height, width)

    # add background noise
    image += np.random.poisson(n_photons_background, size=image.shape)

    # blur image over coordinates, but not over hyb/channels (dim 0, 1)
    sigma = (0, 0) + point_spread_function
    image.values = gaussian_filter(image, sigma=sigma)

    # correct for camera efficiency
    image *= camera_detection_efficiency + np.random.normal(scale=background_electrons, size=image.shape)

    # mimic analog to digital conversion
    image = (image / graylevel).astype(int).clip(0, 2 ** ad_conversion_bits)

    # find the right dtype for the synthetic intensities
    # TODO remove the intensities
    corrected_signal = select_uint_dtype(image)
    corrected_signal.values = rescale_intensity(corrected_signal.values)
    corrected_signal.values = np.clip(corrected_signal, 0, np.inf)

    return corrected_signal, spot_attributes


# TODO this is also imagestack related
class SyntheticSpotTileProvider:

    def __init__(self,
                 hyb=DEFAULT_NUM_HYB,
                 ch=DEFAULT_NUM_CH,
                 z=DEFAULT_NUM_Z,
                 height=DEFAULT_HEIGHT,
                 width=DEFAULT_WIDTH):
        data, spot_attributes = _synthetic_spots(hyb, ch, z, height, width)
        self.data = data

    def tile(self, hyb: int, ch: int, z: int, *args, **kwargs):
        return self.data.sel(h=hyb, c=ch, z=z)
