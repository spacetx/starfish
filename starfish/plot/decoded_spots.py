from typing import Optional, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from starfish import IntensityTable
from starfish.types import Features
from .style import BACKGROUND_COLORMAP, TARGETS_COLORMAP
from .util import annotate_axis, dpi_correction, remove_axis_frame, set_default_arguments


# TODO use the name of the IntensityTable if it exists for a title
# TODO ambrosejcarr: currently the colormaps for decoded_image and intensities do not align.
# however, it should be possible to fix that.
def decoded_spots(
        intensities: IntensityTable=None,
        decoded_image: Optional[np.ndarray]=None,
        background_image: Optional[np.ndarray]=None,
        background_shape: Optional[Tuple[int, int]]=None,
        ax: Optional[matplotlib.axes.Axes]=None,
        background_kwargs: Optional[dict]=None,
        spots_kwargs: Optional[dict]=None,
        decoded_image_kwargs: Optional[dict]=None,
) -> matplotlib.axes.Axes:
    """Plot decoded spots with a variety of options for background image

    Usage
    -----

    Plotting Spots:

    Provide either
    1. intensities, in which case spots will be plotted based on the estimated
       radius and position for each feature, or
    2. decoded_image, in which case pixels will be colored by their decoded feature

    Plotting Background:

    Provide one of:
    1. background_image, which will plot the image for use as a background. Commonly this could
       include a nuclei or dots image.
    2. background_shape, which will be used to construct a black background against which spots
       will be plotted. If decoded_image is provided as the spots parameter, then background
       arguments can be omitted, since the shape of the background can be inferred from the
       decoded_image.

    Parameters
    ----------
    intensities : Optional[IntensityTable]
        contains spots to plot
    decoded_image : Optional[np.ndarray]
        image, where values are coded to represent features
    background_image :
        an image atop which spots or decoded pixels should be plotted
    background_shape :
        the shape for a blank background. Required when background_image is not provided and
        intensities are passed.
    ax : ax: Optional[matplotlib.axes.Axes]
    background_kwargs, spots_kwargs, decoded_image_kwargs : Optional[dict]
        Keyword arguments to pass to plotting functions for background, spots, and decoded_image

    Returns
    -------
    matplotlib.axes.Axes :
        The axis the plot was constructed in.

    """

    # check that enough information on the image extent has been provided
    if all(p is None for p in (background_image, background_shape, decoded_image)):
        raise ValueError(
            'One of background_image, background_shape, or decoded_image must be provided to '
            'set the pixel size of the plot.'
        )

    if intensities is None and decoded_image is None:
        raise ValueError(
            'One of intensities or decoded_image must be provided to visualize spots '
        )

    # if needed, get an axis to plot in
    if ax is None:
        ax = plt.gca()

    # set backgrund plotting arguments
    background_kwargs = set_default_arguments(background_kwargs, cmap=BACKGROUND_COLORMAP)

    # determine what the background should be
    if background_image is None:
        if decoded_image is not None:
            background_image = np.zeros_like(decoded_image)
        else:
            background_image = np.zeros(background_shape)

    # plot the background
    ax.imshow(background_image, **background_kwargs)

    if intensities is not None:
        all_targets = set(np.unique(intensities[Features.TARGET])) - {'None'}
        color_map = dict(zip(all_targets, np.arange(len(all_targets))))

        x = intensities.x[intensities[Features.TARGET] != 'None'].values
        y = intensities.y[intensities[Features.TARGET] != 'None'].values

        targets = intensities[Features.TARGET][intensities[Features.TARGET] != 'None']
        colors = [color_map[t] for t in targets.values]

        # size in matplotlib is a bit involved. First, get the radii
        size = intensities.radius[intensities[Features.TARGET] != 'None'].values
        # make a DPI correction
        size = size * dpi_correction()
        # scatterplot produces dots of size equal to sqrt(s), so square to get correct radius
        size = size ** 2

        # set spots visualization parameters
        spots_kwargs = set_default_arguments(
            spots_kwargs,
            cmap=TARGETS_COLORMAP,
            alpha=0.6
        )

        ax.scatter(x, y, s=size, c=colors, **spots_kwargs)

    if decoded_image is not None:

        # set decoded parameters
        decoded_image_kwargs = set_default_arguments(
            decoded_image_kwargs,
            alpha=0.5,
            cmap=TARGETS_COLORMAP
        )

        masked_image = np.ma.masked_equal(decoded_image, 0)
        ax.imshow(masked_image, interpolation='none', **decoded_image_kwargs)

    remove_axis_frame(ax)
    annotate_axis(ax, title='Decoded Image')
    return ax
