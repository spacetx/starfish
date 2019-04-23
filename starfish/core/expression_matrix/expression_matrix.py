import xarray as xr

from starfish.core.util.try_import import try_import


class ExpressionMatrix(xr.DataArray):

    """Container for expression data extracted from an IntensityTable

    An ExpressionMatrix is a 2-dimensional ``cells`` x ``genes`` tensor whose values are the
    number of spots observed for each gene observed by the experiment. In addition to the basic
    xarray methods, IntensityTable implements:

    Methods
    -------
    save(filename)
        save the ExpressionMatrix to netCDF

    save_loom(filename)
        save the ExpressionMatrix to loom for use in R or python

    save_anndata(filename)
        save the ExpressionMatrix to AnnData for use in ``Scanpy``

    load(filename)
        load an ExpressionMatrix from netCDF
    """

    def save(self, filename: str) -> None:
        """Save an ExpressionMatrix as a Netcdf File

        Parameters
        ----------
        filename : str
            Name of Netcdf file
        """
        self.to_netcdf(filename)

    @try_import({"loompy"})
    def save_loom(self, filename: str) -> None:
        """Save an ExpressionMatrix as a loom file

        Parameters
        ----------
        filename : str
            Name of loom file
        """
        import loompy

        row_attrs = {k: self['cells'][k].values for k in self['cells'].coords}
        col_attrs = {k: self['genes'][k].values for k in self['genes'].coords}

        loompy.create(filename, self.data, row_attrs, col_attrs)

    @try_import({"anndata"})
    def save_anndata(self, filename: str) -> None:
        """Save an ExpressionMatrix as an AnnData file

        Parameters
        ----------
        filename : str
            Name of AnnData file
        """
        import anndata

        row_attrs = {k: self['cells'][k].values for k in self['cells'].coords}
        col_attrs = {k: self['genes'][k].values for k in self['genes'].coords}
        anndata = anndata.AnnData(self.data, row_attrs, col_attrs)
        anndata.write(filename)

    @classmethod
    def load(cls, filename: str) -> "ExpressionMatrix":
        """load an ExpressionMatrix from Netcdf

        Parameters
        ----------
        filename : str
            File to load

        Returns
        -------
        ExpressionMatrix

        """
        loaded = xr.open_dataarray(filename)
        expression_matrix = cls(
            loaded.data,
            loaded.coords,
            loaded.dims
        )
        return expression_matrix
