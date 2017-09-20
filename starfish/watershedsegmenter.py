from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.measurements as spm
from scipy.ndimage import distance_transform_edt
from showit import image
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from .filters import bin_thresh, bin_open
from .munge import relabel
from .stats import label_to_regions


class WatershedSegmenter:
    def __init__(self, dapi_img, stain_img):
        self.dapi = dapi_img / dapi_img.max()
        self.stain = stain_img / stain_img.max()

        self.dapi_thresholded = None
        self.markers = None
        self.num_cells = None
        self.mask = None
        self.segmented = None

    def segment(self, dapi_thresh, stain_thresh, size_lim, disk_size_markers=None, disk_size_mask=None, min_dist=None):
        min_allowed_size, max_allowed_size = size_lim
        self.dapi_thresholded = self.filter_dapi(dapi_thresh, disk_size_markers)
        self.markers, self.num_cells = self.label_nuclei(min_allowed_size, max_allowed_size, min_dist)
        self.mask = self.watershed_mask(stain_thresh, self.markers, disk_size_mask)
        self.segmented = self.watershed(self.markers, self.mask)
        return self.segmented

    def filter_dapi(self, dapi_thresh, disk_size):
        dapi_filt = bin_thresh(self.dapi, dapi_thresh)
        if disk_size is not None:
            dapi_filt = bin_open(dapi_filt, disk_size)
        return dapi_filt

    def label_nuclei(self, min_allowed_size, max_allowed_size, min_dist=None):

        if min_dist is None:
            markers, num_objs = spm.label(self.dapi_thresholded)
        else:
            markers, num_objs = self._unclump(min_dist)

        min_allowed_area = min_allowed_size ** 2
        max_allowed_area = max_allowed_size ** 2

        areas = spm.sum(np.ones(self.dapi_thresholded.shape),
                        markers,
                        np.array(range(0, num_objs + 1), dtype=np.int32))

        area_image = areas[markers]

        markers[area_image <= min_allowed_area] = 0
        markers[area_image >= max_allowed_area] = 0

        markers_reduced, num_objs = relabel(markers)

        return markers_reduced, num_objs

    def _unclump(self, min_dist):
        im = self.dapi_thresholded
        distance = distance_transform_edt(im)
        local_maxi = peak_local_max(distance, labels=im, indices=False, min_distance=min_dist)
        markers, num_objs = spm.label(local_maxi)
        labels_ws = watershed(-distance, markers, mask=im)
        return labels_ws, num_objs

    def watershed_mask(self, stain_thresh, markers, disk_size):
        st = self.stain >= stain_thresh
        watershed_mask = np.logical_or(st, markers > 0)
        if disk_size is not None:
            watershed_mask = bin_open(watershed_mask, disk_size)
        return watershed_mask

    def watershed(self, markers, watershed_mask):
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
        plt.figure(figsize=figsize)

        plt.subplot(321)
        image(self.dapi, ax=plt.gca(), size=20, bar=True)
        plt.title('DAPI')

        plt.subplot(322)
        image(self.stain, ax=plt.gca(), size=20, bar=True)
        plt.title('Stain')

        plt.subplot(323)
        image(self.dapi_thresholded, bar=True, ax=plt.gca())
        plt.title('DAPI Thresholded')

        plt.subplot(324)
        image(self.mask, bar=True, ax=plt.gca())
        plt.title('Watershed Mask')

        plt.subplot(325)
        marker_regions = label_to_regions(self.markers)
        im = marker_regions.mask(background=[0.9, 0.9, 0.9], dims=self.markers.shape, stroke=None, cmap='rainbow')
        image(im, size=20, ax=plt.gca())
        plt.title('Found: {} cells'.format(self.num_cells))

        plt.subplot(326)
        segmented_regions = label_to_regions(self.segmented)
        im = segmented_regions.mask(background=[0.9, 0.9, 0.9], dims=self.segmented.shape, stroke=None, cmap='rainbow')
        image(im, size=20, ax=plt.gca())
        plt.title('Segmented Cells')

        return plt.gca()
