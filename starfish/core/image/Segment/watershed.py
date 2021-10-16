from typing import Optional, Tuple

import numpy as np
from showit import image
from skimage.morphology import disk

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.morphology import Filter, Merge
from starfish.core.morphology.Binarize import ThresholdBinarize
from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.morphology.Filter.areafilter import AreaFilter
from starfish.core.morphology.Filter.min_distance_label import MinDistanceLabel
from starfish.core.morphology.Filter.structural_label import StructuralLabel
from starfish.core.morphology.Segment import WatershedSegment
from starfish.core.types import Axes, FunctionSource, Levels, Number
from ._base import SegmentAlgorithm


class Watershed(SegmentAlgorithm):
    """
    Implements watershed segmentation of cells.

    Algorithm is seeded by nuclei image. Binary segmentation mask is computed from a maximum
    projection of spots across C and R, which is subsequently thresholded.

    This function wraps :py:func:`skimage.segmentation.watershed`

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
    See also: :doc:`skimage:auto_examples/segmentation/plot_watershed`

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
        size_lim = (10, 10000)
        disk_size_markers = None
        disk_size_mask = None

        self._segmentation_instance = _WatershedSegmenter(primary_images, nuclei)
        return self._segmentation_instance.segment(
            self.nuclei_threshold, self.input_threshold, size_lim, disk_size_markers,
            disk_size_mask, self.min_distance
        )

    def show(self, figsize: Tuple[int, int] = (10, 10)) -> None:
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

        self.markers: Optional[BinaryMaskCollection] = None
        self.num_cells: Optional[int] = None
        self.mask: Optional[BinaryMaskCollection] = None
        self.segmented: Optional[BinaryMaskCollection] = None

    def segment(
            self,
            nuclei_thresh: Number,
            stain_thresh: Number,
            size_lim: Tuple[int, int],
            disk_size_markers: Optional[int] = None,  # TODO ambrosejcarr what is this doing?
            disk_size_mask: Optional[int] = None,  # TODO ambrosejcarr what is this doing?
            min_dist: Optional[int] = None
    ) -> BinaryMaskCollection:
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
        BinaryMaskCollection :
            binary mask collection where each cell is a mask.
        """
        min_allowed_size, max_allowed_size = size_lim
        self.binarized_nuclei = self.filter_nuclei(nuclei_thresh, disk_size_markers)
        # label thresholded nuclei image
        if min_dist is None:
            labeled_masks = StructuralLabel().run(self.binarized_nuclei)
        else:
            labeled_masks = MinDistanceLabel(min_dist, 1).run(self.binarized_nuclei)

        area_filter = AreaFilter(min_area=min_allowed_size, max_area=max_allowed_size)
        self.markers = area_filter.run(labeled_masks)
        self.num_cells = len(self.markers)
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
            markers: BinaryMaskCollection,
            disk_size: Optional[int],
    ) -> BinaryMaskCollection:
        """Create a watershed mask that is the union of the spot intensities above stain_thresh and
        a marker image generated from nuclei

        Parameters
        ----------
        stain_thresh : Number
            threshold to apply to the stain image
        markers : BinaryMaskCollection
            markers image generated from nuclei
        disk_size : Optional[int]
            if provided, execute a morphological opening operation over the thresholded stain image

        Returns
        -------
        BinaryMaskCollection :
            watershed mask
        """
        thresholded_stain = ThresholdBinarize(stain_thresh).run(self.stain)
        markers_and_stain = Merge.SimpleMerge().run([thresholded_stain, markers])
        watershed_mask = Filter.Reduce(
            "logical_or",
            lambda shape: np.zeros(shape=shape, dtype=bool)
        ).run(markers_and_stain)
        if disk_size is not None:
            disk_img = disk(disk_size)
            watershed_mask = Filter.Map(
                "morphology.binary_open",
                disk_img,
                module=FunctionSource.skimage
            ).run(watershed_mask)

        return watershed_mask

    def watershed(
            self,
            markers: BinaryMaskCollection,
            watershed_mask: BinaryMaskCollection,
    ) -> BinaryMaskCollection:
        """Run watershed on the thresholded primary_images max projection

        Parameters
        ----------
        markers : BinaryMaskCollection
            markers image generated from nuclei
        watershed_mask : BinaryMaskCollection
            Mask array. only points at which mask == True will be labeled in the output.

        Returns
        -------
        BinaryMaskCollection :
            binary mask collection where each cell is a mask.
        """
        assert len(watershed_mask) == 1

        binarizer = WatershedSegment(connectivity=np.ones((1, 3, 3), dtype=bool))

        return binarizer.run(
            self.stain,
            markers,
            watershed_mask,
        )

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
        image(
            self.mask.to_label_image().xarray.squeeze(Axes.ZPLANE.value).values,
            bar=False,
            ax=plt.gca(),
        )
        plt.title('Watershed Mask')

        plt.subplot(325)
        image(
            self.markers.to_label_image().xarray.squeeze(Axes.ZPLANE.value).values,
            size=20,
            cmap=plt.cm.nipy_spectral,
            ax=plt.gca(),
        )
        plt.title('Found: {} cells'.format(self.num_cells))

        plt.subplot(326)
        image(
            self.segmented.to_label_image().xarray.squeeze(Axes.ZPLANE.value).values,
            size=20,
            cmap=plt.cm.nipy_spectral,
            ax=plt.gca(),
        )
        plt.title('Segmented Cells')

        return plt.gca()
