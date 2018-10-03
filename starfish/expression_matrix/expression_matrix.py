import anndata
import loompy
import xarray as xr


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

    load_loom(filename)
        load an ExpressionMatrix from loom

    load_anndata(filename)
        load an ExpressionMatrix from AnnData


    Examples
    --------
    # TODO ambrosejcarr write examples

    """

    def to_anndata(self) -> anndata.AnnData:
        """convert ExpressionMatrix to a scanpy-compatible AnnData object"""
        row_attrs = {k: self['cells'][k].values for k in self['cells'].coords}
        col_attrs = {k: self['genes'][k].values for k in self['genes'].coords}
        return anndata.AnnData(self.data, row_attrs, col_attrs)

    def save(self, filename: str) -> None:
        """Save an ExpressionMatrix as a Netcdf File

        Parameters
        ----------
        filename : str
            Name of Netcdf file

        """
        self.to_netcdf(filename)

    def save_loom(self, filename: str) -> None:
        """Save an ExpressionMatrix as a loom file

        Parameters
        ----------
        filename : str
            Name of loom file

        """
        row_attrs = {k: self['cells'][k].values for k in self['cells'].coords}
        col_attrs = {k: self['genes'][k].values for k in self['genes'].coords}

        loompy.create(filename, self.data, row_attrs, col_attrs)

    def save_anndata(self, filename: str) -> None:
        """Save an ExpressionMatrix as an AnnData file

        Parameters
        ----------
        filename : str
            Name of AnnData file

        """
        self.to_anndata().write(filename)

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

    def load_anndata(self, filename: str) -> "ExpressionMatrix":
        """load an ExpressionMatrix from AnnData

        Parameters
        ----------
        filename : str
            File to load

        Returns
        -------
        ExpressionMatrix

        """
        adata = anndata.read(filename)
        coordinates = {k: ("cells", adata.obs[k].values) for k in adata.obs.columns}
        coordinates.update({k: ("genes", adata.var[k].values) for k in adata.var.columns})
        return ExpressionMatrix(
            data=adata.X,
            dims=("cells", "genes"),
            coords=coordinates
        )

    def load_loom(self, filename: str) -> "ExpressionMatrix":
        """load an ExpressionMatrix from loom

        Parameters
        ----------
        filename : str
            File to load

        Returns
        -------
        ExpressionMatrix

        """
        with loompy.connect(filename) as ds:
            coordinates = {k: ("cells", ds.ra[k]) for k in ds.ra.keys()}
            coordinates.update({k: ("genes", ds.ca[k]) for k in ds.ca.keys()})
            return ExpressionMatrix(
                data=ds[:, :],
                dims=("cells", "genes"),
                coords=coordinates
            )
