from typing import Optional
import warnings

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from starfish.stats import feature_trace_magnitudes
from starfish.types import Features
from .util import annotate_axis


def spot_area_histogram(
        intensities,
        ax: Optional[matplotlib.axes.Axes],
        bins=25,
        log=True,
        **kwargs,
) -> matplotlib.axes.Axes:
    """

    Parameters
    ----------
    intensities
    ax
    bins
    log
    kwargs

    Returns
    -------

    """
    ax = ax if ax is not None else plt.gca()
    area = (intensities.radius * np.pi) ** 2
    ax.hist(area, bins=bins, log=log, **kwargs)
    annotate_axis(
        ax,
        title='spot area distribution',
        ylabel='number of spots',
        xlabel='area'
    )
    return ax


def spot_distance_histogram(
        intensities,
        ax: Optional[matplotlib.axes.Axes],
        bins=25,
        **kwargs,
) -> matplotlib.axes.Axes:
    """

    Parameters
    ----------
    intensities
    ax
    bins
    kwargs

    Returns
    -------

    """
    ax = ax if ax is not None else plt.gca()
    ax.hist(intensities[Features.DISTANCE].values, bins=bins, **kwargs)
    annotate_axis(
        ax,
        title='distance to nearest\ncode distribution',
        ylabel='number of features',
        xlabel='distance to nearest code'
    )
    return ax


def barcode_magnitude_histogram(
        intensities,
        ax: Optional[matplotlib.axes.Axes],
        log=True,
        bins=100,
        **kwargs,
) -> matplotlib.axes.Axes:
    """

    Parameters
    ----------
    intensities
    ax
    log
    bins
    kwargs

    Returns
    -------

    """
    ax = ax if ax is not None else plt.gca()
    magnitudes = feature_trace_magnitudes(intensities, norm_order=2)

    ax.hist(magnitudes, log=log, bins=bins, **kwargs)
    annotate_axis(
        ax,
        title='barcode magnitude\ndistribution',
        ylabel='number of pixels',
        xlabel='barcode magnitude'
    )

    return ax


# TODO ambrosejcarr: /do we want equal sized axes
def compare_copy_number(
        intensities,
        other: pd.Series,
        ax: Optional[matplotlib.axes.Axes]=None,
        **kwargs,
) -> matplotlib.axes.Axes:
    """

    Parameters
    ----------
    intensities
    other
    ax
    kwargs

    Returns
    -------

    """

    ax = ax if ax is not None else plt.gca()
    targets, counts = np.unique(intensities[Features.TARGET].values, return_counts=True)
    this = pd.Series(counts, index=targets)
    tmp = pd.concat([this, other], join='outer', axis=1, sort=True)
    tmp.columns = ['result', 'comparison']

    # this seaborn function does not accept an axis, but we can make sure it draws to the right
    # place by setting the current axis
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        sns.regplot('comparison', 'result', data=tmp, ax=ax, **kwargs)

    # calculate the correlation coefficient
    # mask nans
    tmp = tmp.dropna()
    r, _ = pearsonr(tmp['result'], tmp['comparison'])

    plt.text(0.1, 0.85, f'r = {r:.3}', transform=ax.transAxes, fontsize=14)
    return ax


