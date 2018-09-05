from typing import Any, Dict, Optional

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt


# TODO ambrosejcarr: need to scale font size with figure size
# TODO ambrosejcarr: need a function to equalize plot axis sizes

def set_default_arguments(dictionary: Optional[Dict]=None, **kwargs) -> Dict[str, Any]:
    """
    Non-overwriting kwargs setter. Sets arguments in an optional dictionary only if they are not
    already present

    Parameters
    ----------
    dictionary : Dict
        dictionary to set keyword arguments in. Created if not provided
    kwargs : Dict
        keywords to add to the dictionary

    Returns
    -------
    Dict :
        Dictionary containing keywords

    """
    dictionary = {} if dictionary is None else dictionary
    for key, value in kwargs.items():
        if key not in dictionary:
            dictionary[key] = value
    return dictionary


def dpi_correction(figure: Optional[matplotlib.figure.Figure]=None) -> float:
    """
    matplotlib.scatter size correction for DPI to ensure that points have constant size across
    visible scales

    Parameters
    ----------
    figure : Optional[matplotlib.figure.Figure]
        Figure in which to scale scatterplot size. If not provided, defaults to current figure.

    Returns
    -------
    float :
        multiplier for matplotlib.plot.scatter markersize kwarg to ensure correct size in pixels

    """
    if figure is None:
        figure = plt.gcf()
    # 72 is the default pixel:point ratio for matplotlib
    point_multiplier = 72. / figure.dpi
    return point_multiplier


def remove_axis_frame(axis: Optional[matplotlib.axes.Axes]=None) -> matplotlib.axes.Axes:
    """Remove the spines and ticks from an axis.

    Parameters
    ----------
    axis : Optional[matplotlib.axes.Axes]
        Optional matplotlib axis. Defaults to current axis if not provided

    Returns
    -------

    """
    axis = axis if axis is not None else plt.gca()
    axis.set_axis_off()
    return axis


def annotate_axis(
        axis: Optional[matplotlib.axes.Axes]=None,
        title: Optional[str]=None,
        xlabel: Optional[str]=None,
        ylabel: Optional[str]=None,
        **kwargs
) -> matplotlib.axes.Axes:
    """Convenience parameter to set labels on an axis

    Parameters
    ----------
    axis : Optional[matplotlib.axes.Axes]
        Optional axis. Defaults to current Axis.
    title : Optional[str]
        title to set
    xlabel : Optional[str]
        xlabel to set
    ylabel : Optional[str]
        ylabel to set
    kwargs : Dict
        keyword arguments to pass to ALL annotations

    Returns
    -------

    """
    axis = axis if axis is not None else plt.gca()
    if title is not None:
        axis.set_title(title, **kwargs)
    if xlabel is not None:
        axis.set_xlabel(xlabel, **kwargs)
    if ylabel is not None:
        axis.set_ylabel(ylabel, **kwargs)
    return axis


def equalize_axes_extents(axis: Optional[matplotlib.axes.Axes]=None) -> matplotlib.axes.Axes:
    """set the x and y axes to the greater extent of both axes"""
    axis = axis if axis is not None else plt.gca()
    x_min, x_max = axis.get_ylim()
    y_min, y_max = axis.get_xlim()
    max_ = max(x_max, y_max)
    min_ = min(x_min, x_max)
    axis.set_xlim(min_, max_)
    axis.set_ylim(min_, max_)
    return axis
