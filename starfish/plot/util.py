from typing import Any, Dict, Optional

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt


# TODO ambrosejcarr: need to scale font size with figure size
# TODO ambrosejcarr: need a function to equalize plot axis sizes

def set_default_arguments(dictionary: Optional[Dict]=None, **kwargs) -> Dict[str, Any]:
    dictionary = {} if dictionary is None else dictionary
    for key, value in kwargs.items():
        if key not in dictionary:
            dictionary[key] = value
    return dictionary


def dpi_correction(figure: Optional[matplotlib.figure.Figure]=None) -> float:
    if figure is None:
        figure = plt.gcf()
        point_multiplier = 72. / figure.dpi
        return point_multiplier


def remove_axis_frame(axis: Optional[matplotlib.axes.Axes]=None) -> matplotlib.axes.Axes:
    axis = axis if axis is not None else plt.gca()
    axis.set_axis_off()
    return axis


def annotate_axis(
        axis: Optional[matplotlib.axes.Axes]=None,
        title: Optional[str]=None,
        xlabel: Optional[str]=None,
        ylabel: Optional[str]=None,
) -> matplotlib.axes.Axes:
    """

    Parameters
    ----------
    axis
    title
    xlabel
    ylabel

    Returns
    -------

    """
    axis = axis if axis is not None else plt.gca()
    if title is not None:
        axis.set_title(title)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    return axis
