.. _DecodedSpotsSpecification:

DecodedSpots
================
The :py:class:`DecodedSpots` is a 2-dimensional tabular data structure where each record represents
a spot, and each record contains, at minimum, columns that specify the `x` and `y` coordinates of
the spot in physical space and the genes that it targets.

Additional columns are not validated, however there are several optional columns with standard
names. If those data are available, methods that take :py:class:`DecodedSpots` objects may make
use of them, so it is in users interests to name those columns appropriately.

Required Columns
----------------

These columns must be present for an object to be constructed.

:code:`target (str):` the gene target for the spot

:code:`x (float):` the x coordinate of the spot in physical space

:code:`y (float):` the y coordinate of the spot in physical space

Optional Columns
----------------

These columns are not validated, but have special meaning to :py:class:`DecodedSpots`

:code:`z (float):` the z coordinate of the spot in physical space

:code:`target_probability (float):` the quality of the decoding, or probability that the spot is
associated with the listed target.

:code:`cell (int):` the identifier of the cell that contains this spot.

:code:`cell_probability (float):` the quality of a cell association, or probability that the spot
is associated with the listed cell.

Implementation
--------------
Starfish Implements the :py:class:`DecodedSpots` as a wrapper for the :code:`pd.DataFrame` object

Serialization
-------------
:py:class:`DecodedSpots` defines methods to save and load the spot file as a csv file.
