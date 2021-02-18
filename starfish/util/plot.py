import itertools
from typing import Any, cast, Hashable, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap

from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.types import Axes, Features


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

    Parameters
    ----------
    image_stack : ImageStack
        imagestack from which to extract a 2-d image for plotting
    sel : Optional[Mapping[Axes, Union[int, tuple]]]
        Optional, but only if image_stack is already of shape (1, 1, 1, y, x). Selector to pass
        ImageStack.sel, Selects the (y, x) plane to be plotted.
    ax :
        Axes to plot on. If not passed, defaults to the current axes.
    title : Optional[str]
        Title to assign the Axes being plotted on.
    kwargs :
        additional keyword arguments to pass to plt.imshow

    """
    if ax is None:
        ax = plt.gca()

    if sel is not None:
        image_stack = image_stack.sel(sel)

    if title is not None:
        ax.set_title(title)

    # verify imagestack is 2d before trying to plot it
    data: xr.DataArray = image_stack.xarray.squeeze()
    if set(data.sizes.keys()).intersection({Axes.CH, Axes.ROUND, Axes.ZPLANE}):
        raise ValueError(f"image_stack must be a 2d (x, y) array, not {data.sizes}")

    # set imshow default kwargs
    if "cmap" not in kwargs:
        kwargs["cmap"] = plt.cm.gray

    ax.imshow(data, **kwargs)
    ax.axis("off")


def intensity_histogram(
    image_stack: ImageStack,
    sel: Optional[Mapping[Axes, Union[int, tuple]]] = None,
    ax=None,
    title: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot the 1-d intensity histogram of linearized image_stack.

    Parameters
    ----------
    image_stack : ImageStack
        imagestack containing intensities
    sel : Optional[Mapping[Axes, Union[int, tuple]]]
        Optional, Selector to pass ImageStack.sel that will restrict the histogram construction to
        the specified subset of image_stack.
    ax :
        Axes to plot on. If not passed, defaults to the current axes.
    title : Optional[str]
        Title to assign the Axes being plotted on.
    kwargs :
        additional keyword arguments to pass to plt.hist

    """
    if ax is None:
        ax = plt.gca()

    if sel is not None:
        image_stack = image_stack.sel(sel)

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

    Parameters
    ----------
    image_stack : ImageStack
        imagestack from which to extract a 2-d image for plotting
    intensities : IntensityTable
        contains spots to overlay on ImageStack.
    sel : Optional[Mapping[Axes, Union[int, tuple]]]
        Optional, but only if image_stack is already of shape (1, 1, 1, y, x). Selector to pass
        ImageStack.sel, Selects the (y, x) plane to be plotted. Will also be used to reduce the
        spots from intensities.
    ax :
        Axes to plot on. If not passed, defaults to the current axes.
    title : Optional[str]
        Title to assign the Axes being plotted on.
    imshow_kwargs : Optional[Mapping[str, Any]]
        additional keyword arguments to pass to imshow
    scatter_kwargs : Optional[Mapping[str, Any]]
        additional keyword arguments to pass to scatter
    """
    if ax is None:
        ax = plt.gca()

    if sel is not None:
        image_stack = image_stack.sel(sel)

        # subset the intensities if needed
        intensity_keys = (Axes.ROUND, Axes.CH, Axes.ZPLANE)
        intensity_sel: Mapping[Hashable, Any] = {
            x.value: sel[x] for x in intensity_keys if x in sel
        }
        if intensity_sel:
            intensities = cast(IntensityTable, intensities.sel(intensity_sel))

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

    # reset the axes limits; scatter often extends them.
    ax.set_ylim((0, image_stack.shape[Axes.Y.value]))
    ax.set_xlim((0, image_stack.shape[Axes.X.value]))


def _linear_alpha_cmap(cmap):
    """add linear alpha to an existing colormap"""
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    return ListedColormap(alpha_cmap)


def diagnose_registration(
    imagestack: ImageStack,
    *sels: Optional[Mapping[Axes, Union[int, tuple]]],
    ax=None,
    title: Optional[str] = None,
    **kwargs,
):
    """
    Overlays 2-d images extracted from ImageStack for the purpose of visualizing alignment of
    images from different rounds or channels selected with the `sel` parameter. Up to six images
    can be selected and shown in different colors. The same `Axes.X` and `Axes.Y` indices should
    be used for every Selector.

    Parameters
    ----------
    imagestack : ImageStack
        imagestack from which to extract 2-d images for plotting
    *sels : Optional[Mapping[Axes, Union[int, tuple]]]
        Optional, but only if image_stack is already of shape (1, 1, 1, y, x). Selectors to pass
        ImageStack.sel, Selects the (y, x) planes to be plotted.
    ax :
        Axes to plot on. If not passed, defaults to the current axes.
    title : Optional[str]
        Title to assign the Axes being plotted on.
    kwargs :
        additional keyword arguments to pass to imshow_plane
    """
    if ax is None:
        ax = plt.gca()

    if title is not None:
        ax.set_title(title)

    # add linear alpha to existing colormaps to avoid the "white" color of the
    # map at 0 from clobbering previously plotted images. Functionally this
    # enables high intensity spots to be co-visualized in the same frame.
    cmaps = [
        plt.cm.Blues,
        plt.cm.Reds,
        plt.cm.Greens,
        plt.cm.Purples,
        plt.cm.Greys,
        plt.cm.Oranges
    ]

    alpha_cmap_cycle = itertools.cycle([_linear_alpha_cmap(cm) for cm in cmaps])

    for sel, cmap in zip(sels, alpha_cmap_cycle):
        imshow_plane(imagestack, sel, ax=ax, cmap=cmap, **kwargs)
