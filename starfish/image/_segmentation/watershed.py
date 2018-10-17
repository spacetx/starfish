from typing import Optional, Tuple

import numpy as np
import regional
import scipy.ndimage.measurements as spm
from scipy.ndimage import distance_transform_edt
from showit import image
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from starfish.image._filter.util import bin_open, bin_thresh
from starfish.imagestack.imagestack import ImageStack
from starfish.munge import relabel
from starfish.stats import label_to_regions
from starfish.types import Indices, Number
from ._base import SegmentationAlgorithmBase


class Watershed(SegmentationAlgorithmBase):

    def __init__(
        self,
        dapi_threshold: Number,
        input_threshold: Number,
        min_distance: int,
        **kwargs
    ) -> None:
        """Implements watershed segmentation of cells seeded from a DAPI image the the point clouds

        Watershed segmentation proceeds by constructing basins that extend from the DAPI image
        and the point cloud constructed from a maximum projection across rounds and channels. It
        constructs a mask that extends from the basins to prevent non-cellular regions from being
        included.

        Parameters
        ----------
        dapi_threshold : Number
            threshold to apply to dapi image
        input_threshold : Number
            threshold to apply to stain image
        min_distance : int
            minimum distance before object centers in a provided dapi image are considered single
            nuclei

        """
        self.dapi_threshold = dapi_threshold
        self.input_threshold = input_threshold
        self.min_distance = min_distance
        self._segmentation_instance: Optional[_WatershedSegmenter] = None

    @classmethod
    def _add_arguments(cls, group_parser) -> None:
        group_parser.add_argument(
            "--dapi-threshold", default=.16, type=float, help="DAPI threshold")
        group_parser.add_argument(
            "--input-threshold", default=.22, type=float, help="Input threshold")
        group_parser.add_argument(
            "--min-distance", default=57, type=int, help="Minimum distance between cells")

    def run(self, primary_images: ImageStack, nuclei: ImageStack) -> regional.many:
        """Segments nuclei in 2-d using a nuclei ImageStack

        Primary images are used to expand the nuclear mask, but only in cases where there are
        densely detected points surrounding the nuclei.

        Parameters
        ----------
        primary_images : ImageStack
            contains primary image data
        nuclei : ImageStack
            contains nuclei image data

        Returns
        -------
        regional.many :
            regional object containing segmentation information
        """

        # create a 'stain' for segmentation
        stain = np.mean(primary_images.max_proj(Indices.CH, Indices.Z), axis=0)
        stain = stain / stain.max()

        # TODO make these parameterizable or determine whether they are useful or not
        size_lim = (10, 10000)
        disk_size_markers = None
        disk_size_mask = None

        nuclei = nuclei.max_proj(Indices.ROUND, Indices.CH, Indices.Z)
        self._segmentation_instance = _WatershedSegmenter(nuclei, stain)
        cells_labels = self._segmentation_instance.segment(
            self.dapi_threshold, self.input_threshold, size_lim, disk_size_markers, disk_size_mask,
            self.min_distance
        )

        regions = label_to_regions(cells_labels)

        return regions

    def show(self, figsize: Tuple[int, int]=(10, 10)) -> None:
        if isinstance(self._segmentation_instance, _WatershedSegmenter):
            self._segmentation_instance.show(figsize=figsize)
        else:
            raise RuntimeError('Run segmentation before attempting to show results.')


class _WatershedSegmenter:
    def __init__(self, dapi_img: np.ndarray, stain_img: np.ndarray) -> None:
        """Implements watershed segmentation of cells seeded from a DAPI image the the point clouds

        Watershed segmentation proceeds by constructing basins that extend from the DAPI image
        and the point cloud constructed from a maximum projection across rounds and channels. It
        constructs a mask that extends from the basins to prevent non-cellular regions from being
        included.

        Parameters
        ----------
        dapi_img : np.ndarray[np.float32]
            nuclei image
        stain_img : np.ndarray[np.float32]
            stain image
        """
        self.dapi = dapi_img / dapi_img.max()
        self.stain = stain_img / stain_img.max()

        self.dapi_thresholded: Optional[np.ndarray] = None  # dtype: bool
        self.markers = None
        self.num_cells: Optional[int] = None
        self.mask = None
        self.segmented = None

    def segment(
            self,
            dapi_thresh: Number,
            stain_thresh: Number,
            size_lim: Tuple[int, int],
            disk_size_markers: Optional[int]=None,  # TODO ambrosejcarr what is this doing?
            disk_size_mask: Optional[int]=None,  # TODO ambrosejcarr what is this doing?
            min_dist: Optional[Number]=None
    ) -> np.ndarray:
        """Execute watershed cell segmentation.

        Parameters
        ----------
        dapi_thresh, stain_thresh : Number
            Threshold for the dapi and stain images. All pixels with intensities above this size
            will be considered part of the objects in the respective images
        size_lim : Tuple[int, int]
            min and max allowable size for nuclei objects in the dapi_image
        disk_size_markers : Optional[int]
            if provided ...
        disk_size_mask : Optional[int]
            if provided ...
        min_dist : Optional[int]
            if provided, nuclei within this distance of each other are combined into single
            objects.

        Returns
        -------
        np.ndarray[int32] :
            label image with same size and shape as self.dapi_img
        """
        min_allowed_size, max_allowed_size = size_lim
        self.dapi_thresholded = self.filter_dapi(dapi_thresh, disk_size_markers)
        self.markers, self.num_cells = self.label_nuclei(
            self.dapi_thresholded,
            min_allowed_size, max_allowed_size, min_dist
        )
        self.mask = self.watershed_mask(stain_thresh, self.markers, disk_size_mask)
        self.segmented = self.watershed(self.markers, self.mask)
        return self.segmented

    def filter_dapi(self, dapi_thresh: float, disk_size: Optional[int]) -> np.ndarray:
        """Threshold the dapi image at dapi_thresh.

        Parameters
        ----------
        dapi_thresh : float
            Threshold the dapi image at this value
        disk_size : int
            if passed, execute a binary opening of the filtered image

        Returns
        -------
        np.ndarray[bool] :
            thresholded image
        """
        dapi_filt = bin_thresh(self.dapi, dapi_thresh)
        if disk_size is not None:
            dapi_filt = bin_open(dapi_filt, disk_size)
        return dapi_filt

    def label_nuclei(
        self,
        dapi_thresholded: np.ndarray,
        min_allowed_size: int,
        max_allowed_size: int,
        min_dist: Optional[Number]=None
    ) -> Tuple[np.ndarray, int]:
        """Construct a labeled nuclei image, which will be combined with the point cloud to seed
        the watershed

        Parameters
        ----------
        dapi_thresholded : np.ndarray
            thresholded dapi image
        min_allowed_size : int
            minimum allowable thresholded nuclei size
        max_allowed_size : int
            maximum allowable nuclei size

        Returns
        -------
        np.ndarray :
            labeled nuclei, excluding those whose size is outside the area boundaries

        """

        # label thresholded nuclei image
        if min_dist is None:
            markers, num_objs = spm.label(dapi_thresholded)
        else:
            markers, num_objs = self._unclump(min_dist)

        # TODO dganguli: does it really make sense to assume a square area?
        min_allowed_area = min_allowed_size ** 2
        max_allowed_area = max_allowed_size ** 2

        # spm.sum sums the values of an array by label. This counts the pixels in each object
        areas = spm.sum(
            np.ones(dapi_thresholded.shape),
            markers,
            np.array(range(0, num_objs + 1), dtype=np.int32)
        )

        # each label value is replaced by its area
        area_image = areas[markers]

        # areas are used to mask values that are outside the allowable sizes
        markers[area_image <= min_allowed_area] = 0
        markers[area_image >= max_allowed_area] = 0

        markers_reduced, num_objs = relabel(markers)

        return markers_reduced, num_objs

    def _unclump(self, min_dist: Number) -> Tuple[np.ndarray, int]:
        """
        Run watershed on the thresholded basin image, restricted to basins at least min_dist apart

        Functionally, this reproduces the thresholded dapi image with overlapping nuclei merged.

        Parameters
        ----------
        min_dist : int
            minimum distance between watershed basins

        """
        im: np.ndarray = self.dapi_thresholded

        # calculates the distance of every pixel to the nearest background (0) point
        distance: np.ndarray = distance_transform_edt(im)  # dtype: np.float64

        # boolean array marking local maxima, excluding any maxima within min_dist
        local_maxi: np.ndarray = peak_local_max(
            distance, labels=im, indices=False, min_distance=min_dist
        )

        # label the maxima for watershed
        markers, num_objs = spm.label(local_maxi)

        # run watershed, using the distances in the thresholded image as basins.
        # Uses the original image as a mask, preventing any background pixels from being labeled
        labels_ws: np.ndarray = watershed(-distance, markers, mask=im)
        return labels_ws, num_objs

    def watershed_mask(self, stain_thresh: Number, markers: np.ndarray, disk_size: Optional[int]):
        """Create a watershed mask that is the union of the spot intensities above stain_thresh and
        a marker image generated from nuclei

        Parameters
        ----------
        stain_thresh : Number
            threshold to apply to the stain image
        markers : np.ndarray[bool]
            markers for the stain_image
        disk_size : Optional[int]
            if provided, execute a morphological opening operation over the thresholded stain image

        Returns
        -------
        np.ndarray[bool] :
            thresholded stain image

        """
        st = self.stain >= stain_thresh
        watershed_mask: np.ndarray = np.logical_or(st, markers > 0)  # dtype bool
        if disk_size is not None:
            watershed_mask = bin_open(watershed_mask, disk_size)
        return watershed_mask

    def watershed(self, markers: np.ndarray, watershed_mask: np.ndarray) -> np.ndarray:
        """Run watershed on the thresholded primary_images max projection

        Parameters
        ----------
        markers : np.ndarray[np.int64]
            an array marking the basins with the values to be assigned in the label matrix.
            Zero means not a marker.
        watershed_mask : np.ndarray[bool]
            Mask array. only points at which mask == True will be labeled in the output.

        Returns
        -------
        np.ndarray[np.int32] :
            labeled image, each segment has a unique integer value
        """
        img = 1 - self.stain

        res = watershed(image=img,
                        markers=markers,
                        connectivity=np.ones((3, 3), bool),
                        mask=watershed_mask
                        )

        return res

    def to_regions(self):
        regions = label_to_regions(self.segmented)
        return regions

    def show(self, figsize=(10, 10)):
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)

        plt.subplot(321)
        image(self.dapi, ax=plt.gca(), size=20, bar=True)
        plt.title('DAPI')

        plt.subplot(322)
        image(self.stain, ax=plt.gca(), size=20, bar=True)
        plt.title('Stain')

        plt.subplot(323)
        image(self.dapi_thresholded, bar=False, ax=plt.gca())
        plt.title('DAPI Thresholded')

        plt.subplot(324)
        image(self.mask, bar=False, ax=plt.gca())
        plt.title('Watershed Mask')

        plt.subplot(325)
        marker_regions = label_to_regions(self.markers)
        im = marker_regions.mask(
            background=[0.9, 0.9, 0.9],
            dims=self.markers.shape,
            stroke=None,
            cmap='rainbow'
        )
        image(im, size=20, ax=plt.gca())
        plt.title('Found: {} cells'.format(self.num_cells))

        plt.subplot(326)
        segmented_regions = label_to_regions(self.segmented)
        im = segmented_regions.mask(
            background=[0.9, 0.9, 0.9],
            dims=self.segmented.shape,
            stroke=None,
            cmap='rainbow'
        )
        image(im, size=20, ax=plt.gca())
        plt.title('Segmented Cells')

        return plt.gca()
