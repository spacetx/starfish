from typing import Mapping, Optional, Tuple

import numpy as np
from showit import image
from skimage.morphology import disk, watershed

from starfish.core.image.Filter.util import bin_open
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.morphology import Filter
from starfish.core.morphology.Binarize import ThresholdBinarize
from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.morphology.Filter.areafilter import AreaFilter
from starfish.core.morphology.Filter.min_distance_label import MinDistanceLabel
from starfish.core.morphology.Filter.structural_label import StructuralLabel
from starfish.core.morphology.label_image import LabelImage
from starfish.core.types import ArrayLike, Axes, Coordinates, FunctionSource, Levels, Number
from ._base import SegmentAlgorithm


class Watershed(SegmentAlgorithm):
    """
    Implements watershed segmentation of cells.

    Algorithm is seeded by nuclei image. Binary segmentation mask is computed from a maximum
    projection of spots across C and R, which is subsequently thresholded.

    This function wraps :py:func:`skimage.morphology.watershed`

    Parameters
    ----------
    nuclei_threshold : Number
        threshold to apply to nuclei image
    input_threshold : Number
        threshold to apply to stain image
    min_distance : int
        minimum distance before object centers in a provided nuclei image are considered single
        nuclei

    Notes
    -----
    Watershed: http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

    """

    def __init__(
        self,
        nuclei_threshold: Number,
        input_threshold: Number,
        min_distance: int,
    ) -> None:

        self.nuclei_threshold = nuclei_threshold
        self.input_threshold = input_threshold
        self.min_distance = min_distance
        self._segmentation_instance: Optional[_WatershedSegmenter] = None

    def run(
            self,
            primary_images: ImageStack,
            nuclei: ImageStack,
            *args
    ) -> BinaryMaskCollection:
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
        masks : BinaryMaskCollection
           binary masks segmenting each cell
        """
        # TODO make these parameterizable or determine whether they are useful or not
        size_lim = (10, 10000)
        disk_size_markers = None
        disk_size_mask = None

        self._segmentation_instance = _WatershedSegmenter(primary_images, nuclei)
        label_image_array = self._segmentation_instance.segment(
            self.nuclei_threshold, self.input_threshold, size_lim, disk_size_markers,
            disk_size_mask, self.min_distance
        )

        # we max-projected and squeezed the Z-plane so label_image.ndim == 2
        physical_ticks: Mapping[Coordinates, ArrayLike[Number]] = {
            coord: nuclei.xarray.coords[coord.value].data
            for coord in (Coordinates.Y, Coordinates.X)
        }

        return BinaryMaskCollection.from_label_array_and_ticks(
            label_image_array,
            None,
            physical_ticks,
            None,  # TODO: (ttung) this should really be logged.
        )

    def show(self, figsize: Tuple[int, int]=(10, 10)) -> None:
        if isinstance(self._segmentation_instance, _WatershedSegmenter):
            self._segmentation_instance.show(figsize=figsize)
        else:
            raise RuntimeError('Run segmentation before attempting to show results.')


class _WatershedSegmenter:
    def __init__(self, primary_images: ImageStack, nuclei: ImageStack) -> None:
        """Implements watershed segmentation of cells seeded from a nuclei image

        Algorithm is seeded by a nuclei image. Binary segmentation mask is computed from a maximum
        projection of spots across C and R, which is subsequently thresholded.

        Parameters
        ----------
        primary_images : ImageStack
            primary hybridization images
        nuclei : ImageStack
            nuclei image
        """
        # create a 'stain' for segmentation
        mp = primary_images.reduce({Axes.CH, Axes.ZPLANE}, func="max")
        self.stain = mp.reduce({
            Axes.ROUND},
            func="mean",
            level_method=Levels.SCALE_BY_IMAGE)

        self.nuclei_mp_scaled = nuclei.reduce(
            {Axes.ROUND, Axes.CH, Axes.ZPLANE},
            func="max",
            level_method=Levels.SCALE_BY_IMAGE,
        )

        self.nuclei_thresholded: Optional[np.ndarray] = None  # dtype: bool
        self.markers: Optional[LabelImage] = None
        self.num_cells: Optional[int] = None
        self.mask = None
        self.segmented = None

    def segment(
            self,
            nuclei_thresh: Number,
            stain_thresh: Number,
            size_lim: Tuple[int, int],
            disk_size_markers: Optional[int]=None,  # TODO ambrosejcarr what is this doing?
            disk_size_mask: Optional[int]=None,  # TODO ambrosejcarr what is this doing?
            min_dist: Optional[int] = None
    ) -> np.ndarray:
        """Execute watershed cell segmentation.

        Parameters
        ----------
        nuclei_thresh, stain_thresh : Number
            Threshold for the nuclei and stain images. All pixels with intensities above this size
            will be considered part of the objects in the respective images
        size_lim : Tuple[int, int]
            min and max allowable size for nuclei objects in the nuclei_image
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
            label image with same size and shape as self.nuclei_img
        """
        min_allowed_size, max_allowed_size = size_lim
        self.binarized_nuclei = self.filter_nuclei(nuclei_thresh, disk_size_markers)
        # label thresholded nuclei image
        if min_dist is None:
            labeled_masks = StructuralLabel().run(self.binarized_nuclei)
        else:
            labeled_masks = MinDistanceLabel(min_dist, 1).run(self.binarized_nuclei)

        area_filter = AreaFilter(min_area=min_allowed_size, max_area=max_allowed_size)
        filtered_masks = area_filter.run(labeled_masks)
        self.num_cells = len(filtered_masks)
        self.markers = filtered_masks.to_label_image()
        self.mask = self.watershed_mask(stain_thresh, self.markers, disk_size_mask)
        self.segmented = self.watershed(self.markers, self.mask)
        return self.segmented

    def filter_nuclei(self, nuclei_thresh: float, disk_size: Optional[int]) -> BinaryMaskCollection:
        """Binarize the nuclei image using a thresholded binarizer and perform morphological binary
        opening.

        Parameters
        ----------
        nuclei_thresh : float
            Threshold the nuclei image at this value
        disk_size : int
            if passed, execute a binary opening of the filtered image

        Returns
        -------
        BinaryMaskCollection :
            mask collection with one mask, which is
        """
        nuclei_binarized = ThresholdBinarize(nuclei_thresh).run(self.nuclei_mp_scaled)
        if disk_size is not None:
            disk_img = disk(disk_size)
            nuclei_binarized = Filter.Map(
                "morphology.binary_open",
                disk_img,
                module=FunctionSource.skimage
            ).run(nuclei_binarized)

        # should only produce one binary mask.
        assert len(nuclei_binarized) == 1
        return nuclei_binarized

    def watershed_mask(
            self,
            stain_thresh: Number,
            markers: LabelImage,
            disk_size: Optional[int],
    ) -> np.ndarray:
        """Create a watershed mask that is the union of the spot intensities above stain_thresh and
        a marker image generated from nuclei

        Parameters
        ----------
        stain_thresh : Number
            threshold to apply to the stain image
        markers : LabelImage
            markers image generated from nuclei
        disk_size : Optional[int]
            if provided, execute a morphological opening operation over the thresholded stain image

        Returns
        -------
        np.ndarray[bool] :
            thresholded stain image

        """
        st = self.stain._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE) >= stain_thresh
        markers_any = (markers.xarray > 0).values.squeeze(axis=0)
        watershed_mask: np.ndarray = np.logical_or(st, markers_any)  # dtype bool
        if disk_size is not None:
            watershed_mask = bin_open(watershed_mask, disk_size)
        return watershed_mask

    def watershed(self, markers: LabelImage, watershed_mask: np.ndarray) -> np.ndarray:
        """Run watershed on the thresholded primary_images max projection

        Parameters
        ----------
        markers : LabelImage
            markers image generated from nuclei
        watershed_mask : np.ndarray[bool]
            Mask array. only points at which mask == True will be labeled in the output.

        Returns
        -------
        np.ndarray[np.int32] :
            labeled image, each segment has a unique integer value
        """
        img = 1 - self.stain._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)

        res = watershed(image=img,
                        markers=markers.xarray.values.squeeze(axis=0),
                        connectivity=np.ones((3, 3), bool),
                        mask=watershed_mask
                        )

        return res

    def show(self, figsize=(10, 10)):
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)

        plt.subplot(321)
        nuclei_numpy = self.nuclei_mp_scaled._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE)
        image(nuclei_numpy, ax=plt.gca(), size=20, bar=True)
        plt.title('Nuclei')

        plt.subplot(322)
        image(
            self.stain._squeezed_numpy(Axes.ROUND, Axes.CH, Axes.ZPLANE),
            ax=plt.gca(), size=20, bar=True)
        plt.title('Stain')

        plt.subplot(323)
        image(
            self.binarized_nuclei.uncropped_mask(0).squeeze(Axes.ZPLANE.value).values,
            bar=False,
            ax=plt.gca(),
        )
        plt.title('Nuclei Thresholded')

        plt.subplot(324)
        image(self.mask, bar=False, ax=plt.gca())
        plt.title('Watershed Mask')

        plt.subplot(325)
        image(
            self.markers.xarray.squeeze(Axes.ZPLANE.value).values,
            size=20,
            cmap=plt.cm.nipy_spectral,
            ax=plt.gca())
        plt.title('Found: {} cells'.format(self.num_cells))

        plt.subplot(326)
        image(self.segmented, size=20, cmap=plt.cm.nipy_spectral, ax=plt.gca())
        plt.title('Segmented Cells')

        return plt.gca()
