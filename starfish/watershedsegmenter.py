from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.measurements as spm
from centrosome.cpmorphology import relabel
from showit import image
from skimage.morphology import watershed


class WatershedSegmenter:
    def __init__(self, dapi_img, stain_img):
        self.dapi = dapi_img / dapi_img.max()
        self.stain = stain_img / stain_img.max()

        self.dapi_thresholded = None
        self.markers = None
        self.num_cells = None
        self.mask = None
        self.segmented = None

    def segment(self, dapi_thresh, stain_thresh, min_allowed_size, max_allowed_size):
        self.threshold_dapi(dapi_thresh)
        self.markers, self.num_cells = self.label_nuclei(min_allowed_size, max_allowed_size)
        self.mask = self.watershed_mask(stain_thresh, self.markers)
        self.segmented = self.watershed(self.markers, self.mask)
        return self.segmented

    def threshold_dapi(self, dapi_thresh):
        self.dapi_thresholded = self.dapi >= dapi_thresh

    def label_nuclei(self, min_allowed_size, max_allowed_size):
        markers, num_objs = spm.label(self.dapi_thresholded)

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

    def watershed_mask(self, stain_thresh, markers):
        st = self.stain >= stain_thresh
        watershed_mask = np.logical_or(st, markers > 0)
        return watershed_mask

    def watershed(self, markers, watershed_mask):
        img = 1 - self.stain

        res = watershed(image=img,
                        markers=markers,
                        connectivity=np.ones((3, 3), bool),
                        mask=watershed_mask
                        )

        return res

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
        image(self.markers, cmap=plt.cm.spectral, bar=True, ax=plt.gca())
        plt.title('Found: {} cells'.format(self.num_cells))

        plt.subplot(326)
        image(self.segmented, cmap=plt.cm.spectral, bar=True, ax=plt.gca())
        plt.title('Cel'.format(self.num_cells))

        return plt.gca()
