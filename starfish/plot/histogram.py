from typing import Optional, Sequence

import matplotlib.axes
import matplotlib.pyplot as plt

from starfish.types import Number
from .util import annotate_axis


def _plot_threshold_indicator(x_location: Number, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    """plots a vertical dotted line to represent a threshold applied to the histogram"""
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=x_location, ymin=ymin, ymax=ymax, linestyles='--', colors='r')
    ax.set_ylim(ymin, ymax)  # reset the axis. Matplotlib expands when using vlines
    return ax


def histogram(
        data_vector: Sequence,
        ax: Optional[matplotlib.axes.Axes]=None,
        bins: int=25,
        threshold: Optional[Number]=None,
        title: Optional[str]='',
        xlabel: Optional[str]='',
        ylabel: Optional[str]='',
        log=True,
        **kwargs,
) -> matplotlib.axes.Axes:
    """Plot the distribution of spot areas

    Parameters
    ----------
    data_vector: Sequence
        data vector to be summarized
    ax : Optional[matplotlib.axes.Axes]
        Axis to plot in. If not provided, defaults to current axis.
    bins :
        Number of bins. Default 25.
    log : bool
        Whether to plot the y-axis of the histogram in log scale.
    threshold: Optional[Number]
        If provided, plot a vertical line indicating where a threshold was selected for a given
        assay
    title, xlabel, ylabel : Optional[str]
        Labels to add to the title, x, and y labels of the plot
    kwargs : Dict
        additional keyword arguments to pass to matplotlib.pyplot.hist

    Returns
    -------
    matplotlib.axes.Axes :
        The axis containing the plot

    """
    ax = ax if ax is not None else plt.gca()
    ax.hist(data_vector, bins=bins, log=log, **kwargs)
    annotate_axis(
        ax,
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
    )
    if threshold is not None:
        _plot_threshold_indicator(threshold, ax)

    return ax
