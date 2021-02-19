import __main__
import base64
import warnings
from collections import OrderedDict
from io import BytesIO
from typing import Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from packaging.version import parse as parse_version

from starfish.core.imagestack.dataorder import AXES_DATA, N_AXES
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.morphology.binary_mask import BinaryMaskCollection
from starfish.core.types import Axes, Features

try:
    from napari import Viewer as NapariViewer

    class Viewer(NapariViewer):

        def _get_png(self):
            from imageio import imsave

            image = self.screenshot(canvas_only=False)
            with BytesIO() as file_obj:
                imsave(file_obj, image, format='png')
                file_obj.seek(0)
                png = file_obj.read()
            return png

        def _repr_html_(self):

            png = self._get_png()
            data = base64.b64encode(png).decode('utf-8')

            return (
                "<div>"
                f"<img class='screenshot' src='data:image/png;base64, {data}' />"
                "</div>"
            )

except ImportError:
    Viewer = None  # type: ignore


NAPARI_VERSION = "0.3.4"  # when changing this, update docs in display
INTERACTIVE = not hasattr(__main__, "__file__")


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
        warnings.simplefilter("ignore", RuntimeWarning)
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
    coords[:, 0] = np.tile(np.tile(range(n_rounds), n_channels), n_features)
    coords[:, 1] = np.tile(np.repeat(range(n_channels), n_rounds), n_features)
    coords[:, 2] = np.repeat(intensity_table[Features.AXIS][Axes.ZPLANE.value].values, code_length)
    coords[:, 3] = np.repeat(intensity_table[Features.AXIS][Axes.Y.value].values, code_length)
    coords[:, 4] = np.repeat(intensity_table[Features.AXIS][Axes.X.value].values, code_length)

    sizes = np.repeat(intensity_table.radius.values, code_length)
    rc = np.zeros((sizes.shape[0], 2), dtype=int)
    z = sizes[:, np.newaxis]
    yx = np.tile(sizes[:, np.newaxis], (1, 2))
    sizes = np.concatenate((rc, yx, z), axis=1)

    return coords, sizes


def display(
        stack: Optional[ImageStack] = None,
        spots: Optional[IntensityTable] = None,
        masks: Optional[BinaryMaskCollection] = None,
        viewer: Optional[Viewer] = None,
        project_axes: Optional[Set[Axes]] = None,
        mask_intensities: float = 0.,
        radius_multiplier: int = 1,
        z_multiplier: float = 1
):
    """
    Display an image stack, detected spots, and masks using
    `napari <https://github.com/napari/napari>`.

    Parameters
    ----------
    stack : ImageStack
        ImageStack to display
    spots : IntensityTable
        IntensityTable containing spot information that was generated from the submitted stack.
    masks : BinaryMaskCollection
        Segmentation instance masks used to annotate the submitted stack.
    viewer : napari.Viewer
        Napari viewer to append the ImageStack and/or spots to. If None, creates a new viewer.
        Note: appending is only supported in interactive environments.
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
    z_multiplier : int
        Multiplies the radius of the spots in z, to account for anisotropy.

    Examples
    --------
    1. Display a stack to evaluate a filtering result. Just pass any ImageStack!

    >>> from starfish import display
    >>> display(stack)

    2. Display spots of a single-molecule FISH experiment: smFISH will produce IntensityTables where
    most values are np.NaN. These will be masked automatically, so passing an ImageStack +
    IntensityTable will display spots in the rounds and channels that they are detected

    >>> from starfish import display
    >>> display(stack, intensities)

    3. Diagnose spot calls within each round and channel of a coded experiment. A user might want to
    evaluate if spot calling is failing for a specific round/channel pair. To accomplish this,
    pass the intensity threshold used by the spot called to eliminate sub-threshold spots from
    channels. The user can additionally filter the IntensityTable to further mask additional spots
    (any np.NaN value will not be displayed)

    >>> from starfish import display
    >>> mask_intensities = 0.3  # this was the spot calling intensity threshold
    >>> display(stack, intensities, mask_intensities=mask_intensities)

    4. Evaluate spot calls across rounds and channels by visualizing spots on a max projection of
    The rounds and channels.

    >>> from starfish import display, Axes
    >>> display(stack, intensities, project_axes={Axes.CH, Axes.ROUND})

    5. Compare the image before (raw_stack) and after (filtered_stack) filtering by displaying
    two stacks in the same Viewer.

    >>> from starfish import display
    >>> viewer = display(raw_stack)
    >>> viewer = display(stack=filtered_stack, viewer=viewer)

    Notes
    -----
    - To use in ipython, use the `%gui qt` magic.
    - napari axes are labeled with the ImageStack axis names
    - Requires napari 0.3.4: use `pip install starfish[napari]`
      to install all necessary requirements
    """
    if stack is None and spots is None and masks is None:
        raise TypeError("expected a stack, spots, or masks; got nothing")

    try:
        import napari
    except ImportError:
        raise ImportError(f"Requires at least napari {NAPARI_VERSION}. "
                          "Run `pip install starfish[napari]` "
                          "to install the necessary requirements.")

    try:
        version = napari.__version__
    except Exception as e:
        raise RuntimeError("Could not identify napari version") from e

    if parse_version(version) < parse_version(NAPARI_VERSION):
        raise ValueError(f"Incorrect version {version} of napari installed."
                         "Run `pip install starfish[napari]` "
                         "to install the necessary requirements.")

    # instantiate the napari viewer
    if viewer is None:
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])

        # Get the list of axis labels
        axis_labels = []
        for ix in range(N_AXES):
            for axis_name, axis_data in AXES_DATA.items():
                if ix == axis_data.order:
                    axis_labels.append(axis_name.value)
                    break
            else:
                raise Exception(f"No AXES_DATA entry for the {ix}th axis")
        axis_labels.append(Axes.Y.value)
        axis_labels.append(Axes.X.value)

        viewer = Viewer(axis_labels=axis_labels)
        new_viewer = True
    elif isinstance(viewer, Viewer):
        new_viewer = False
    else:
        raise TypeError("viewer must be a napari.Viewer or None")

    if stack is not None:
        if project_axes is not None:
            stack = stack.reduce(project_axes, func="max")

        viewer.add_image(stack.xarray.values,
                         rgb=False,
                         name="stack")

    if spots is not None:
        if project_axes is not None:
            spots = _max_intensity_table_maintain_dims(spots, project_axes)

        # TODO ambrosejcarr guard rails:
        # 1. verify that x, y, z fit inside the imagestack (weird projections)
        # 2. warn if whole tiles are missing spot calls (possible miss-use)

        coords, sizes = _spots_to_markers(spots)

        # mask low-intensity values
        mask = _mask_low_intensity_spots(spots, mask_intensities)

        if not np.sum(mask):
            warnings.warn(f"No spots passed provided intensity threshold of {mask_intensities}")
        else:
            coords = coords[mask]
            sizes = sizes[mask]

            # adjust z-size
            sizes[:, 4] *= z_multiplier

            viewer.add_points(
                coords,
                face_color="red",
                edge_color="red",
                symbol="ring",
                size=sizes * radius_multiplier,
                n_dimensional=True,
                name="spots"
            )

    if masks:
        viewer.add_labels(masks.to_label_image().label_image,
                          name="masks")

    if new_viewer and not INTERACTIVE:
        app.exec_()  # create blocking process to persist windows

    return viewer
