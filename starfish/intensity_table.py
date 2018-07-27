from itertools import product
from typing import Union, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from starfish.constants import Indices, Features
from starfish.munge import dataframe_to_multiindex


class IntensityTable(xr.DataArray):
    """3 dimensional container for spot/pixel features extracted from image data

    An IntensityTable is comprised of each feature's intensity across channels and imaging
    rounds, where features are typically spots or pixels. This forms an
    (n_feature, n_channel, n_round) tensor implemented as an xarray.DataArray object.
    In addition to the basic xarray methods, IntensityTable implements:

    Constructors
    -------
    empty_intensity_table  creates an IntensityTable with all intensities equal to zero
    from_spot_data         creates an IntensityTable from a 3d array and a spot attributes dataframe
    synthetic_intensities  creates an IntensityTable with synthetic spots, given a codebook

    Methods
    -------
    save                   save the IntensityTable to netCDF
    load                   load an IntensityTable from netCDF

    Examples
    --------
    >>> from starfish.util.synthesize import SyntheticData
    >>> sd = SyntheticData(n_ch=3, n_round=4, n_codes=2)
    >>> codes = sd.codebook()
    >>> sd.intensities(codebook=codes)
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

    """

    # constant that stores the attr key for the shape of ImageStack
    IMAGE_SHAPE = 'image_shape'

    @classmethod
    def empty_intensity_table(
            cls, spot_attributes: pd.MultiIndex, n_ch: int, n_round: int,
            image_shape: Tuple[int, int, int]
    ) -> "IntensityTable":
        """Create an empty intensity table with pre-set axis whose values are zero

        Parameters
        ----------
        spot_attributes : pd.MultiIndex
            MultiIndex containing spot metadata. Must contain the values specifid in Constants.X,
            Y, Z, and RADIUS.
        n_ch : int
            number of channels measured in the imaging experiment
        n_round : int
            number of imaging rounds measured in the imaging experiment
        image_shape : Tuple[int, int, int]
            the shape (z, y, x) of the image from which features will be extracted

        Returns
        -------
        IntensityTable :
            empty IntensityTable

        """
        cls._verify_spot_attributes(spot_attributes)
        channel_index = np.arange(n_ch)
        round_index = np.arange(n_round)
        data = np.zeros((spot_attributes.shape[0], n_ch, n_round))
        dims = (Features.AXIS, Indices.CH.value, Indices.ROUND.value)
        attrs = {cls.IMAGE_SHAPE: image_shape}

        intensity_table = cls(
            data=data, coords=(spot_attributes, channel_index, round_index), dims=dims,
            attrs=attrs
        )

        return intensity_table

    @staticmethod
    def _verify_spot_attributes(spot_attributes: pd.MultiIndex) -> None:
        """Run some checks on spot attributes"""
        if not isinstance(spot_attributes, pd.MultiIndex):
            raise ValueError(
                f'spot attributes must be a pandas MultiIndex, not {type(spot_attributes)}.')

        required_attributes = {Features.Z, Features.Y, Features.X}
        missing_attributes = required_attributes.difference(spot_attributes.names)
        if missing_attributes:
            raise ValueError(
                f'Missing spot_attribute levels in provided MultiIndex: {missing_attributes}. '
                f'The following levels are required: {required_attributes}.')

    @classmethod
    def from_spot_data(
            cls, intensities: Union[xr.DataArray, np.ndarray], spot_attributes: pd.MultiIndex,
            image_shape: Tuple[int, int, int],
            *args, **kwargs) -> "IntensityTable":
        """Table to store image feature intensities and associated metadata

        Parameters
        ----------
        intensities : np.ndarray[Any]
            intensity data
        spot_attributes : pd.MultiIndex
            Name(s) of the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            are taken from ``coords`` (if possible) and otherwise default to
            ``['dim_0', ... 'dim_n']``.
        image_shape : Tuple[int, int, int]
            the shape of the image (z, y, x) from which the features were extracted
        args :
            additional arguments to pass to the xarray constructor
        kwargs :
            additional keyword arguments to pass to the xarray constructor

        Returns
        -------
        IntensityTable :
            IntensityTable containing data from passed ndarray, annotated by spot_attributes

        """

        if len(intensities.shape) != 3:
            raise ValueError(
                f'intensities must be a (features * ch * round) 3-d tensor. Provided intensities '
                f'shape ({intensities.shape}) is invalid.')

        cls._verify_spot_attributes(spot_attributes)

        coords = (
            (Features.AXIS, spot_attributes),
            (Indices.CH.value, np.arange(intensities.shape[1])),
            (Indices.ROUND.value, np.arange(intensities.shape[2]))
        )

        dims = (Features.AXIS, Indices.CH.value, Indices.ROUND.value)

        attrs = {cls.IMAGE_SHAPE: image_shape}

        return cls(intensities, coords, dims, attrs=attrs, *args, **kwargs)

    def save(self, filename: str) -> None:
        """Save an IntensityTable as a Netcdf File

        Parameters
        ----------
        filename : str
            Name of Netcdf file

        """
        # TODO when https://github.com/pydata/xarray/issues/1077 (support for multiindex
        # serliazation) is merged, remove this reset_index() call and simplify load, below
        self.reset_index('features').to_netcdf(filename)

    @classmethod
    def load(cls, filename: str) -> "IntensityTable":
        """load an IntensityTable from Netcdf

        Parameters
        ----------
        filename : str
            File to load

        Returns
        -------
        IntensityTable

        """
        loaded = xr.open_dataarray(filename)
        intensity_table = cls(
            loaded.data,
            loaded.coords,
            loaded.dims
        )
        return intensity_table.set_index(
            features=list(intensity_table[Features.AXIS].coords.keys()))

    def show(self, background_image: np.ndarray) -> None:
        """show spots on a background image"""
        raise NotImplementedError

    @classmethod
    def synthetic_intensities(
            cls, codebook, num_z: int=12, height: int=50, width: int=40, n_spots=10,
            mean_fluor_per_spot=200, mean_photons_per_fluor=50
    ) -> "IntensityTable":
        """Create an IntensityTable containing synthetic spots with random locations

        Parameters
        ----------
        codebook : Codebook
            starfish codebook object
        num_z :
            number of z-planes to use when localizing spots
        height :
            y dimension of each synthetic plane
        width :
            x dimension of each synthetic plane
        n_spots :
            number of spots to generate
        mean_fluor_per_spot :
            mean number of fluorophores per spot
        mean_photons_per_fluor :
            mean number of photons per fluorophore.

        Returns
        -------
        IntensityTable

        """

        # TODO nsofroniew: right now there is no jitter on x-y positions of the spots
        z = np.random.randint(0, num_z, size=n_spots)
        y = np.random.randint(0, height, size=n_spots)
        x = np.random.randint(0, width, size=n_spots)
        r = np.empty(n_spots)
        r.fill(np.nan)  # radius is a function of the point-spread gaussian size

        names = [Features.Z, Features.Y, Features.X, Features.SPOT_RADIUS]
        spot_attributes = pd.MultiIndex.from_arrays([z, y, x, r], names=names)

        # empty data tensor
        data = np.zeros(shape=(n_spots, *codebook.shape[1:]))

        targets = np.random.choice(
            codebook.coords[Features.TARGET], size=n_spots, replace=True)
        expected_bright_locations = np.where(codebook.loc[targets])

        # create a binary matrix where "on" spots are 1
        data[expected_bright_locations] = 1

        # add physical properties of fluorescence
        data *= np.random.poisson(mean_photons_per_fluor, size=data.shape)
        data *= np.random.poisson(mean_fluor_per_spot, size=data.shape)

        image_shape = (num_z, height, width)

        intensities = cls.from_spot_data(data, spot_attributes, image_shape=image_shape)
        intensities[Features.TARGET] = (Features.AXIS, targets)

        return intensities

    @classmethod
    def from_image_stack(cls, image_stack, crop_x: int=0, crop_y: int=0, crop_z: int=0) -> "IntensityTable":
        """Generate an IntensityTable from all the pixels in the ImageStack

        Parameters
        ----------
        crop_x : int
            number of pixels to crop from both top and bottom of x
        crop_y : int
            number of pixels to crop from both top and bottom of y
        crop_z : int
            number of pixels to crop from both top and bottom of z
        image_stack : ImageStack
            ImageStack containing pixels to be treated as intensities

        Returns
        -------
        IntensityTable
            IntensityTable containing one intensity per pixel (across channels and rounds)

        """

        # verify the image is large enough to crop
        assert crop_z * 2 < image_stack.shape['z']
        assert crop_y * 2 < image_stack.shape['y']
        assert crop_x * 2 < image_stack.shape['x']

        zmin = crop_z
        ymin = crop_y
        xmin = crop_x
        zmax = image_stack.shape['z'] - crop_z
        ymax = image_stack.shape['y'] - crop_y
        xmax = image_stack.shape['x'] - crop_x
        data = image_stack.numpy_array.transpose(2, 3, 4, 1, 0)  # (z, y, x, ch, round)

        # crop and reshape imagestack to create IntensityTable data
        cropped_data = data[zmin:zmax, ymin:ymax, xmin:xmax, :, :]
        # (pixels, ch, round)
        intensity_data = cropped_data.reshape(-1, image_stack.num_chs, image_stack.num_rounds)

        # IntensityTable pixel coordinates
        z = np.arange(zmin, zmax)
        y = np.arange(ymin, ymax)
        x = np.arange(xmin, xmax)

        pixel_coordinates = pd.DataFrame(
            data=np.array(list(product(z, y, x))),
            columns=['z', 'y', 'x']
        )
        pixel_coordinates[Features.SPOT_RADIUS] = np.full(
            pixel_coordinates.shape[0], fill_value=np.nan)

        spot_attributes = dataframe_to_multiindex(pixel_coordinates)
        image_size = cropped_data.shape[:3]

        return IntensityTable.from_spot_data(intensity_data, spot_attributes, image_size)

    def mask_low_intensity_features(self, intensity_threshold):
        """return the indices of features that have average intensity below intensity_threshold"""
        mask = np.where(
            self.mean([Indices.CH.value, Indices.ROUND.value]).values < intensity_threshold)[0]
        return mask

    def mask_small_features(self, min_size: int, max_size: int):
        """return the indices of features whose radii are smaller than size_threshold"""
        mask = np.where(self.coords.features[Features.SPOT_RADIUS] < min_size)[0]
        mask |= np.where(self.coords.features[Features.SPOT_RADIUS] > max_size)[0]
        return mask

    def _intensities_from_regions(self, props, reduce_op='max') -> "IntensityTable":
        """turn regions back into intensities by reducing over the labeled area"""
        raise NotImplementedError
