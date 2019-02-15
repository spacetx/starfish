import warnings
from collections import OrderedDict
from typing import Iterable, List, Mapping, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import napari_gui
import numpy as np

from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable
from starfish.types import Axes, Features, Number


def _normalize_axes(axes: Iterable[Union[Axes, str]]) -> List[str]:
    """convert all axes to strings"""

    def _normalize(axes):
        if isinstance(axes, str):
            return axes
        elif isinstance(axes, Axes):
            return axes.value
        else:
            raise TypeError(f"passed axes must be str or Axes objects, not {type(axes)}")

    return list(_normalize(i) for i in axes)


def _max_intensity_table_maintain_dims(
    intensity_table: IntensityTable, dimensions: Set[Axes],
) -> IntensityTable:
    """
    Maximum project an IntensityTable over dimensions, retaining singletons in place of dimensions
    reduced by the max operation.

    For example, xarray.max(Axes.CH.value) would usually produce a 2-d (features, rounds) output.
    This function ensures that the output is instead (features, 1, rounds), by inserting singleton
    dimensions in place of any projected axes.

    Parameters
    ----------
    intensity_table : IntensityTable
        Input intensities
    dimensions : Set[Axes]
        Dimensions to project

    Returns
    -------
    IntensityTable :
        full-dimensionality (3-d) IntensityTable

    """
    str_dimensions = _normalize_axes(dimensions)
    initial_dimensions = OrderedDict(intensity_table.sizes)
    projected_intensities = intensity_table.max(str_dimensions)
    expanded_intensities = projected_intensities.expand_dims(str_dimensions)
    return expanded_intensities.transpose(*tuple(initial_dimensions.keys()))


def _mask_low_intensity_spots(
    intensity_table: IntensityTable,
    intensity_threshold: float
) -> np.ndarray:
    """
    returns a flattened boolean mask (c order flattening) that is True where
    (feature, round, channel) combinations whose intensities are above or equal to intensity
    threshold and where values are not nan
    """
    with warnings.catch_warnings():
        # we expect to have invalid values (np.nan) in IntensityTable, hide the warnings from users
        warnings.simplefilter('ignore', RuntimeWarning)
        # reshape the values into a 1-d vector of length product(features, rounds, channels)
        return np.ravel(intensity_table.values >= intensity_threshold)


def _spots_to_markers(intensity_table: IntensityTable) -> Tuple[np.ndarray, np.ndarray]:
    """
    convert each (r, c) combination for each spot into a 5-tuple (x, y, r, c, z) for plotting
    with napari.
    """
    # get sizes which will be used to create the coordinates that correspond to the spots
    n_rounds = intensity_table.sizes[Axes.ROUND.value]
    n_channels = intensity_table.sizes[Axes.CH.value]
    n_features = intensity_table.sizes[Features.AXIS]
    code_length = n_rounds * n_channels

    # create 5-d coordinates for plotting in (x, y, round. channel, z)
    n_markers = np.product(intensity_table.shape)
    coords = np.zeros((n_markers, 5), dtype=np.uint16)

    # create the coordinates.
    # X, Y, and Z are repeated once per (r, ch) pair (code_length).
    # the cartesian product of (r, c) are created, and tiled once per feature (n_features)
    # we ensure that (ch, round) cycle in c-order to match the order of the linearized
    # array, used below for masking.
    coords[:, 0] = np.repeat(intensity_table[Features.AXIS][Axes.X.value].values, code_length)
    coords[:, 1] = np.repeat(intensity_table[Features.AXIS][Axes.Y.value].values, code_length)
    coords[:, 2] = np.tile(np.tile(range(n_rounds), n_channels), n_features)
    coords[:, 3] = np.tile(np.repeat(range(n_channels), n_rounds), n_features)
    coords[:, 4] = np.repeat(intensity_table[Features.AXIS][Axes.ZPLANE.value].values, code_length)

    sizes = np.repeat(intensity_table.radius.values, code_length)[:, np.newaxis]
    # must share order with napari, currently (y, x, r, c, z)
    vector_sizes = np.concatenate(
        [
            np.repeat(sizes, 2, axis=1),  # y, x
            np.repeat(np.full_like(sizes, 0), 2, axis=1),  # no size in channels, rounds
            np.repeat(sizes, 1, axis=1),  # z
        ], axis=1
    )

    return coords, vector_sizes

def _plot_markers(
    window,
    spots: Optional[IntensityTable],
    project_axes: Optional[Set[Axes]],
    mask_intensities: float,
    radius_multiplier: Number,
    color: str,
):
    if project_axes is not None:
        spots = _max_intensity_table_maintain_dims(spots, project_axes)

    coords, sizes = _spots_to_markers(spots)

    # mask low-intensity values
    mask = _mask_low_intensity_spots(spots, mask_intensities)

    if not np.sum(mask):
        warnings.warn(f"No spots passed provided intensity threshold of {mask_intensities}")
        return window

    coords = coords[mask, :]
    sizes = sizes[mask, :]

    window.viewer.add_markers(
        coords=coords, face_color=color, edge_color=color, symbol="o",
        size=sizes * radius_multiplier,
        n_dimensional=True
    )

    return window


def stack(
    stack: ImageStack,
    spots: Optional[IntensityTable] = None,
    project_axes: Optional[Set[Axes]] = None,
    mask_intensities: float = 0.,
    radius_multiplier: int = 1,
    extra_spots: Optional[Mapping[str, Tuple[str, IntensityTable]]] = None,
):
    """
    Displays the image stack using Napari (https://github.com/Napari).
    Can optionally overlay detected spots if the corresponding IntensityTable
    is provided.

    Parameters
    ----------
    stack : ImageStack
        ImageStack to display
    spots : IntensityTable
        IntensityTable containing spot information that was generated from the submitted stack.
    project_axes : Optional[Set[Axes]]
        If provided, both the ImageStack and the Spots will be maximum projected along the
        selected axes. Useful for displaying spots across coded assays where spots may not
        appear in specific rounds or channels.
    mask_intensities : Float
        hide markers that correspond to intensities below this threshold value; note that any
        marker that is np.nan will ALSO be masked, allowing users to pre-mask the intensity table
        (see documentation on IntensityTable.where() for more details) (default 0, no masking)
    radius_multiplier : int
        Multiplies the radius of the displayed spots (default 1, no scaling)
    extra_spots : Optional[Mapping[Tuple[str, IntensityTable]]]
        Optional dict of name: (color, IntensityTable). These spots will be displayed in the same
        manner as the main spots, except a different color. This is useful for contrasting between
        spots that do and do not decode, and can be used to mask spots if the provided color is the
        base color of the cmap (e.g. black, k)

    Examples
    --------

    1. Display a stack to evaluate a filtering result. Just pass any ImageStack!

    >>> import starfish.display
    >>> starfish.display.stack(stack)

    2. Display spots of a single-molecule FISH experiment: smFISH will produce IntensityTables where
    most values are np.NaN. These will be masked automatically, so passing an ImageStack +
    IntensityTable will display spots in the rounds and channels that they are detected

    >>> import starfish.display
    >>> starfish.display.stack(stack, intensities)

    3. Diagnose spot calls within each round and channel of a coded experiment. A user might want to
    evaluate if spot calling is failing for a specific round/channel pair. To accomplish this,
    pass the intensity threshold used by the spot called to eliminate sub-threshold spots from
    channels. The user can additionally filter the IntensityTable to further mask additional spots
    (any np.NaN value will not be displayed)

    >>> import starfish.display
    >>> mask_intensities = 0.3  # this was the spot calling intensity threshold
    >>> starfish.display(stack, intensities, mask_intensities=mask_intensities)

    4. Evaluate spot calls across rounds and channels by visualizing spots on a max projection of
    The rounds and channels.

    >>> import starfish.display
    >>> from starfish import Axes
    >>> starfish.display.stack(stack, intensities, project_axes={Axes.CH, Axes.ROUND})

    Notes
    -----
    - To use in ipython, use the %gui qt5 magic.
    - Napari axes currently cannot be labeled. Until such a time that they can, this function will
      order them by Round, Channel, and Z.
    - Requires napari 0.0.5.1: install starfish using `pip install starfish[napari]` to install all
      necessary requirements
    """
    try:
        import napari_gui
    except ImportError:
        print("Requires napari 0.0.5.1. Run `pip install starfish[napari]` to install the "
              "necessary requirements.")
        return

    if project_axes is not None:
        stack = stack.max_proj(*project_axes)

    # Switch axes to match napari expected order [x, y, round, channel, z]
    reordered_array: np.ndarray = stack.xarray.transpose(
        Axes.Y.value,
        Axes.X.value,
        Axes.ROUND.value,
        Axes.CH.value,
        Axes.ZPLANE.value
    ).values

    # display the imagestack using napari
    window = napari_gui.imshow(reordered_array, multichannel=False)

    if spots is not None:
        _plot_markers(window, spots, project_axes, mask_intensities, radius_multiplier, color="c")

    if extra_spots is not None:
        for name, (color, espots) in extra_spots.items():
            _plot_markers(window, espots, project_axes, mask_intensities, radius_multiplier, color)

    return window
