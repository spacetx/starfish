"""
This module contains a series of utilities for creating two dimensional plots that are useful for
generating documentation and vignettes. We suggest that users leverage :py:func:`starfish.display`
for their plotting needs, as the interactive viewer is better able to handle the array of features
that starfish needs.
"""
from typing import Any, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from starfish import ImageStack, IntensityTable
from starfish.types import Axes, Features


def imshow_plane(
    image_stack: ImageStack,
    sel: Optional[Mapping[Axes, Union[int, tuple]]] = None,
    ax=None,
    title: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plot a single plane of an ImageStack. If passed a selection function (sel), the stack will be
    subset using :py:meth:`ImageStack.sel`. If ax is passed, the function will be plotted in the
    provided axis. Additional kwargs are passed to :py:func:`plt.imshow`
    """
    if ax is None:
        ax = plt.gca()

    if sel is not None:
        image_stack: ImageStack = image_stack.sel(sel)

    if title is not None:
        ax.set_title(title)

    # verify imagestack is 2d before trying to plot it
    data: xr.DataArray = image_stack.xarray.squeeze()
    if set(data.sizes.keys()).intersection({Axes.CH, Axes.ROUND, Axes.ZPLANE}):
        raise ValueError(f"image_stack must be a 2d (x, y) array, not {data.sizes}")

    ax.imshow(data, **kwargs)


def intensity_histogram(
    image_stack: ImageStack,
    sel: Optional[Mapping[Axes, Union[int, tuple]]] = None,
    ax=None,
    title: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot the intensity histogram of image_stack.
    """
    if ax is None:
        ax = plt.gca()

    if sel is not None:
        image_stack: ImageStack = image_stack.sel(sel)

    if title is not None:
        ax.set_title(title)

    data: np.ndarray = np.ravel(image_stack.xarray)
    ax.hist(data, **kwargs)

def overlay_spot_calls(
    image_stack: ImageStack,
    intensities: IntensityTable,
    sel: Optional[Mapping[Axes, Union[int, tuple]]] = None,
    ax=None,
    title: Optional[str] = None,
    imshow_kwargs: Optional[Mapping[str, Any]] = None,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Overlays spot calls atop a 2-d image extracted from ImageStack. Manages sub-selection from the
    IntensityTable and ImageStack based on provided `sel` parameter.
    """
    if ax is None:
        ax = plt.gca()

    if sel is not None:
        image_stack: ImageStack = image_stack.sel(sel)

        # subset the intensities if needed
        intensity_keys = (Axes.ROUND.value, Axes.CH.value, Axes.ZPLANE)
        intensity_sel = {x: sel[x] for x in intensity_keys if x in sel}
        if intensity_sel:
            intensities = intensities.sel(intensity_sel)

    imshow_kwargs = imshow_kwargs if imshow_kwargs else {}
    scatter_kwargs = scatter_kwargs if scatter_kwargs else {}

    # plot background
    imshow_plane(image_stack, sel=sel, ax=ax, title=title, **imshow_kwargs)

    # plot spots
    plt.scatter(
        x=np.asarray(intensities[Axes.X.value]),
        y=np.asarray(intensities[Axes.Y.value]),
        s=np.asarray(intensities[Features.SPOT_RADIUS]),
        c='red',
        **scatter_kwargs,
    )

    # reset the axes limits
    ax.set_ylim((0, image_stack.shape[Axes.Y.value]))
    ax.set_xlim((0, image_stack.shape[Axes.X.value]))
