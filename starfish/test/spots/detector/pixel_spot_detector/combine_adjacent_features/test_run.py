import numpy as np

from starfish.types import Features
from starfish.spots._detector.combine_adjacent_features import (
    CombineAdjacentFeatures,
    combine_adjacent_features
)
from .test_calculate_mean_pixel_traces import labeled_intensities_factory


# TODO add tests about filtering
def test_combine_adjacent_features_no_filtering():
    """
    Combine adjacent features takes a decoded IntensityTable. Ideally this would be based on a
    coherent codebook, but this can be mocked by simply assigning pixel traces different values
    in the IntensityTable.
    """
    intensities, label_image, _ = labeled_intensities_factory()

    # transform the label image into labels by reshaping
    targets = np.ravel(label_image).astype(str)
    intensities[Features.TARGET] = (Features.AXIS, targets)

    min_area = 1
    max_area = 3
    connectivity = 2
    caf = CombineAdjacentFeatures(min_area=min_area, max_area=max_area, connectivity=connectivity)
    results, props = caf.run(intensities)
    results2, props2 = combine_adjacent_features(
        intensities,
        min_area=min_area,
        max_area=max_area,
        connectivity=connectivity
    )

    # verify that results are the same
    assert results.equals(results2)

    # we should have 4 features
    assert results.sizes[Features.AXIS] == 4
