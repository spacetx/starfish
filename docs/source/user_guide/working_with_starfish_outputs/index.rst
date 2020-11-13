.. _working_with_starfish_outputs:

Working with Starfish Outputs
-----------------------------

Starfish's output_formats are serialized as :code:`netcdf` or :code:`csv` files. These files are
easy to work with in both Python_ and R_.

.. _Python: https://www.python.org/
.. _R: https://www.r-project.org/about.html

To work with the :py:class:`IntensityTable` in python, it's as simple as using that object's open_netcdf
command:

.. code-block:: python

    import starfish
    example_netcdf_file: str = "docs/source/_static/example_data_files/decoded.nc"
    intensity_table: starfish.IntensityTable = starfish.IntensityTable.open_netcdf(example_netcdf_file)
    print(intensity_table)

in R_, the ncdf4 library allows the :code:`.nc` archive, which is based on hdf5, to be opened.
It will contain a number of variables, each of which can be accessed by name. Alternative
installation instructions can be accessed here. Alternative installation instructions can be
accessed here_:

.. _here: http://cirrus.ucsd.edu/~pierce/ncdf/

.. code-block:: R

    install.packages("ncdf4")
    library("ncdf4")
    example_netcdf_file <- "docs/source/_static/example_data_files/decoded.nc"
    netcdf_connection <- nc_open(example_netcdf_file)

    # access the z-coordinate vector
    zc <- ncvar_get(netcdf_connection, "zc")
    head(zc)

    # access the 3-dimensional data structure containing intensty information
    # this variable has a special name, the rest are accessible with the string
    # constants you would expect from the starfish python API.
    data <- ncvar_get(netcdf_connection, "__xarray_dataarray_variable__")
    head(data)

To work with the decoded table is even simpler, as they are stored as :code:`.csv` files, and can
be read natively by pandas in Python and natively in R.

Python:

.. code-block:: python

    import pandas as pd
    example_decoded_spots_file: str = "docs/source/_static/example_data_files/decoded.csv"
    table: pd.DataFrame = pd.read_csv(example_decoded_spots_file, index_col=0)
    table.head()

R:

.. code-block:: R

    example_decoded_spots_file <- "docs/source/_static/example_data_files/decoded.csv"
    table <- read.csv(file=example_decoded_spots_file, header=TRUE, sep=',', row.names=1)
    head(table)

Output Formats
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :caption: Contents:

.. toctree::
    IntensityTable/index.rst

.. toctree::
    ExpressionMatrix/index.rst

.. toctree::
    DecodedSpots/index.rst

.. toctree::
    SegmentationMask/index.rst