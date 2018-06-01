from starfish.constants import Indices
from ._base import SegmentationAlgorithmBase


class Watershed(SegmentationAlgorithmBase):
    """
    Implements watershed segmentation.  TODO: (dganguli) FILL IN DETAILS HERE PLS.
    """
    def __init__(self, dapi_threshold, input_threshold, min_distance, **kwargs):
        self.dapi_threshold = dapi_threshold
        self.input_threshold = input_threshold
        self.min_distance = min_distance

    @classmethod
    def get_algorithm_name(cls):
        return "watershed"

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--dapi-threshold", default=.16, type=float, help="DAPI threshold")
        group_parser.add_argument("--input-threshold", default=.22, type=float, help="Input threshold")
        group_parser.add_argument("--min-distance", default=57, type=int, help="Minimum distance between cells")

    def segment(self, stack):
        import numpy as np
        from starfish.stats import label_to_regions
        from starfish.watershedsegmenter import WatershedSegmenter

        # create a 'stain' for segmentation
        # TODO: (ambrosejcarr) is this the appropriate way of dealing with Z in stain generation?
        stain = np.mean(stack.max_proj(Indices.CH, Indices.Z), axis=0)
        stain = stain / stain.max()

        # TODO make these parameterizable or determine whether they are useful or not
        size_lim = (10, 10000)
        disk_size_markers = None
        disk_size_mask = None

        nuclei = stack.auxiliary_images['nuclei'].max_proj(Indices.HYB, Indices.CH, Indices.Z)
        seg = WatershedSegmenter(nuclei, stain)
        cells_labels = seg.segment(
            self.dapi_threshold, self.input_threshold, size_lim, disk_size_markers, disk_size_mask, self.min_distance)

        regions = label_to_regions(cells_labels)

        return regions
