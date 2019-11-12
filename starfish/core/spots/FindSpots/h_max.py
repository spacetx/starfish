from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import distance
from skimage import img_as_float
from skimage.measure import label
from skimage.morphology import h_maxima

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.spots.FindSpots import spot_finding_utils
from starfish.core.types import Axes, Features, SpotAttributes, SpotFindingResults
from ._base import FindSpotsAlgorithm


class HMax(FindSpotsAlgorithm):
    """
    Determine all maxima of the image with height >= h.

    Parameters
    ----------
    h : unsigned integer
        The minimal height of all extracted maxima.
    selem : ndarray, optional
        The neighborhood expressed as an n-D array of 1's and 0's.
        Default is the ball of radius 1 according to the maximum norm
        (i.e. a 3x3 square for 2D images, a 3x3x3 cube for 3D images, etc.)

    Notes
    -----
    https://scikit-image.org/docs/dev/api/skimage.morphology
    """

    def __init__(self, h, selem: np.ndarray=None, is_volume=True, measurement_type='max'):
        self.h = h
        self.selem = selem
        self.is_volume = is_volume
        self.measurement_function = self._get_measurement_function(measurement_type)

    def image_to_spots(self, data_image: Union[np.ndarray, xr.DataArray]) -> SpotAttributes:
        """
        Find spots using a h_maxima algorithm

        Parameters
        ----------
        data_image : Union[np.ndarray, xr.DataArray]
            image containing spots to be detected

        Returns
        -------
        SpotAttributes :
            DataFrame of metadata containing the coordinates, intensity and radius of each spot

        """

        results = h_maxima(image=img_as_float(data_image), h=self.h, selem=self.selem)

        data_image = np.asarray(data_image)

        label_h_max = label(results, neighbors=4)
        # no maxima present in image
        if (label_h_max == np.ones(label_h_max.shape)).all():
            max_mask = np.zeros(data_image.shape)
        else:
            labels = pd.DataFrame(
                data={'labels': np.sort(label_h_max[np.where(label_h_max != 0)])})
            # find duplicates labels (=connected components)
            dup = labels.index[labels.duplicated()].tolist()

            # splitting connected regional maxima to get only one local maxima
            max_mask = np.zeros(data_image.shape)
            max_mask[label_h_max != 0] = 1

            # Compute medoid for connected regional maxima
            for i in range(len(dup)):
                # find coord of points having the same label
                z, r, c = np.where(label_h_max == labels.loc[dup[i], 'labels'])
                meanpoint_x = np.mean(c)
                meanpoint_y = np.mean(r)
                meanpoint_z = np.mean(z)
                dist = [distance.euclidean([meanpoint_z, meanpoint_y, meanpoint_x],
                                           [z[j], r[j], c[j]]) for j in range(len(r))]
                ind = dist.index(min(dist))
                # delete values at ind position.
                z, r, c = np.delete(z, ind), np.delete(r, ind), np.delete(c, ind)
                max_mask[z, r, c] = 0  # set to 0 points != medoid coordinates
        results = max_mask.nonzero()
        results = np.vstack(results).T

        spot_data = pd.DataFrame(
            data={Axes.X.value: results[:, 2],
                  Axes.Y.value: results[:, 1],
                  Axes.ZPLANE.value: results[:, 0],
                  Features.SPOT_RADIUS: 1,
                  Features.SPOT_ID: np.arange(results.shape[0]),
                  Features.INTENSITY: data_image[results[:, 0],
                                                 results[:, 1],
                                                 results[:, 2]]
                  })
        return SpotAttributes(spot_data)

    def run(
            self,
            image_stack: ImageStack,
            reference_image: Optional[ImageStack] = None,
            n_processes: Optional[int] = None,
            *args,
    ) -> SpotFindingResults:
        """
        Find spots in the given ImageStack using a gaussian blob finding algorithm.
        If a reference image is provided the spots will be detected there then measured
        across all rounds and channels in the corresponding ImageStack. If a reference_image
        is not provided spots will be detected _independently_ in each channel. This assumes
        a non-multiplex imaging experiment, as only one (ch, round) will be measured for each spot.

        Parameters
        ----------
        image_stack : ImageStack
            ImageStack where we find the spots in.
        reference_image : xr.DataArray
            (Optional) a reference image. If provided, spots will be found in this image, and then
            the locations that correspond to these spots will be measured across each channel.
        n_processes : Optional[int] = None,
            Number of processes to devote to spot finding.
        """
        spot_finding_method = partial(self.image_to_spots, *args)
        if reference_image:
            data_image = reference_image._squeezed_numpy(*{Axes.ROUND, Axes.CH})
            reference_spots = spot_finding_method(data_image)
            results = spot_finding_utils.measure_intensities_at_spot_locations_across_imagestack(
                data_image=image_stack,
                reference_spots=reference_spots,
                measurement_function=self.measurement_function)
        else:
            spot_attributes_list = image_stack.transform(
                func=spot_finding_method,
                group_by={Axes.ROUND, Axes.CH},
                n_processes=n_processes
            )
            results = SpotFindingResults(imagestack_coords=image_stack.xarray.coords,
                                         log=image_stack.log,
                                         spot_attributes_list=spot_attributes_list)
        return results
