IntensityTable
==============

The :py:class:`IntensityTable` summarizes information about RNA features that are detected in
image-based transcriptomics experiments. The most common features are spots, pixels, or connected
pixels that result from the fluorescence of a probe that has been experimentally attached to an RNA
molecule.

To gather information on the RNA present in a tissue slice, image-based transcriptomics experiments
require the imaging of the same RNA molecules over multiple rounds and across multiple fluorescence
channels. In multiplex assays, the identity of these spots is only determined after measuring the
spot's intensity across all rounds and channels. In sequential assays, each round and channel
identifies all detected RNA molecules for a specific target.

The :py:class:`IntensityTable` stores a summary of the intensity of each feature across each round
and channel, forming a 3-dimensional array with dimensions (feature, round, channel). It also stores
metadata that declare how the features were measured and are defined, and metadata for each
individual feature. Finally, it supports addition of arbitrary metadata for each feature, round,
channel, or on the detection of those features.

Data
----

IntensityTables are stored as three dimensional arrays of shape :code:`(f, r, c)`, where :code:`f`
is the number of detected features in the experiment, :code:`r` is the number of rounds, and
:code:`c` is the number of channels. Entries in the array are floats in the range of :code:`[0, 1]`,
where :code:`1` represents the maximum intensity and :code:`0`, the minimum intensity.

Table Metadata
--------------
The IntensityTable stores some standard metadata that record how features are summarized.

:code:`intensity_measurement_type (string):`

In the case where features are composed of spots or connected pixels, the intensity of each pixel
in the feature must be integrated into a single measurement. The :code:`intensity_measurement_type`
field stores the calculation made to carry out this integration. This entry is filled by the spot
detector. The default value is :code:`max` but other common methods might include :code:`mean` or
:code:`median`.

:code:`area_measurment_type (string):`

This size of each feature can be summarize in different ways that often depend on how they're
detected. For example, it is natural to summarize spots as circles, while pixel-based
approaches can produce irregular features that are difficult to summarize parametrically. This
entry is filled by the spot detector and describes how areas are calculated.


Feature Metadata
----------------
:py:class:`IntensityTable` Features are annotated with several types of metadata. These include the
:code:`x, y, z` position of each feature in pixel and coordinate space and the area of the feature.
Additionally, there are entries that can be filled by a decoder which determines the target that a
spot corresponds to, or a Segmentation mask, which determines which cell an object corresponds to.
Users may add additional metadata columns with names that do not overlap this list as they see fit.
The following metadata are always present on IntensityTables. In some cases, these values may be
set to NaN or null if the IntensityTable has not been decoded or merged with a segmentation result.

:code:`x, y, z (int):` coordinates of the centroid of the feature in pixel space

:code:`xc, yc, zc (float):` coordinates of the centroid of the feature in global coordinates (um)

:code:`area (float):` area of the spot in pixels  # TODO: should this be in um^2?

:code:`cell (int):` the id for the cell that this feature belongs to

:code:`gene (str):` gene name of target RNA, (gene symbol or GENCODE gene ID)

Implementation
--------------
Starfish Implements the :py:class:`ExpressionMatrix` as an :code:`xarray.DataArray` object to take
advantage of `xarray's`_ high performance, flexible metadata storage capabilities, and Serialization
options

.. _`xarray's`: http://xarray.pydata.org/en/stable/

Serialization
-------------
The :py:class:`IntensityTable` can leverage any of the :code:`xarray` serialization features,
including csv, zarr, and netcdf. We choose netcdf as it currently has the strongest support and
interoperability between R and python. Users can load and manipulate :py:class:`IntensityTables`
using R by loading them with the `ncdf4`_ package. In the future, NetCDF serialization may be
deprecated if R gains Zarr support.

.. _ncdf4: https://cran.r-project.org/web/packages/ncdf4/index.html
