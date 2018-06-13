import functools
import json

import numpy as np
import pandas as pd
import xarray as xr

from starfish.constants import Indices, CodebookIndices, IntensityIndices
from starfish.pipeline.features.intensity_table import IntensityTable


class Codebook(xr.DataArray):

    def __init__(self, data, coords, *args, **kwargs):
        super().__init__(data, coords, *args, **kwargs)

    @classmethod
    def from_json(cls, json_codebook, n_hyb, n_ch):
        with open(json_codebook, 'r') as f:
            code_array = json.load(f)
        return cls.from_code_array(code_array, n_hyb, n_ch)

    @classmethod
    def from_code_array(cls, code_array, n_hyb, n_ch):

        for code in code_array:

            if not isinstance(code, dict):
                raise ValueError(f'codebook must be an array of dictionary codes. Found: {code}.')

            # verify all necessary fields are present
            required_fields = {CodebookIndices.CODEWORD.value, CodebookIndices.GENE_NAME.value}
            missing_fields = required_fields.difference(code)
            if missing_fields:
                raise ValueError(
                    f'Each entry of codebook must contain {required_fields}. Missing fields: {missing_fields}')

        # empty codebook
        code_data = cls(
            data=np.zeros((len(code_array), n_ch, n_hyb), dtype=np.uint8),
            coords=(
                pd.Index(
                    [d[CodebookIndices.GENE_NAME.value] for d in code_array],
                    name=CodebookIndices.GENE_NAME.value
                ),
                pd.Index(np.arange(n_ch), name=Indices.CH.value),
                pd.Index(np.arange(n_hyb), name=Indices.HYB.value),
            )
        )

        # fill the codebook
        for code_dict in code_array:
            codeword = code_dict[CodebookIndices.CODEWORD.value]
            gene = code_dict[CodebookIndices.GENE_NAME.value]
            for entry in codeword:
                code_data.loc[gene, entry[Indices.CH.value], entry[Indices.HYB.value]] = entry[
                    CodebookIndices.VALUE.value]

        return code_data

    @staticmethod
    def min_euclidean_distance(observation, codes):
        squared_diff = (codes - observation) ** 2
        code_distances = np.sqrt(squared_diff.sum((Indices.CH, Indices.HYB)))
        # order of codes changes here (automated sorting on the reshaping?)
        return code_distances

    @staticmethod
    def append_multiindex_level(multiindex, data, name):
        """stupid thing necessary because pandas doesn't do this"""
        frame = multiindex.to_frame()
        frame[name] = data
        frame.set_index(name, append=True, inplace=True)
        return frame.index

    def euclidean_decode(self, intensities):
        norm_intensities = intensities.groupby(IntensityIndices.FEATURES.value).apply(lambda x: x / x.sum())
        norm_codes = self.groupby(CodebookIndices.GENE_NAME.value).apply(lambda x: x / x.sum())

        func = functools.partial(self.min_euclidean_distance, codes=norm_codes)
        distances = norm_intensities.groupby(IntensityIndices.FEATURES.value).apply(func)

        qualities = 1 - distances.min(CodebookIndices.GENE_NAME.value)
        closest_code_index = distances.argmin(CodebookIndices.GENE_NAME.value)
        gene_ids = distances.indexes[CodebookIndices.GENE_NAME.value].values[closest_code_index.values]
        with_genes = self.append_multiindex_level(intensities.indexes[IntensityIndices.FEATURES.value], gene_ids, 'gene')
        with_qualities = self.append_multiindex_level(with_genes, qualities, 'quality')

        result = IntensityTable(
            intensities=intensities,
            dims=(IntensityIndices.FEATURES.value, Indices.CH.value, Indices.HYB.value),
            coords=(
                with_qualities,
                intensities.indexes[Indices.CH.value],
                intensities.indexes[Indices.HYB.value]
            )
        )
        return result

    def per_channel_max_decode(self, intensities):

        def view_row_as_element(array):
            nrows, ncols = array.shape
            dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                     'formats': ncols * [array.dtype]}
            return array.view(dtype)

        max_channels = intensities.argmax(Indices.CH.value)
        codes = self.argmax(Indices.CH.value)

        a = view_row_as_element(codes.values.reshape(self.shape[0], -1))
        b = view_row_as_element(max_channels.values.reshape(intensities.shape[0], -1))

        genes = np.empty(intensities.shape[0], dtype=object)
        genes.fill('None')

        for i in np.arange(a.shape[0]):
            genes[np.where(a[i] == b)[0]] = codes['gene_name'][i]
        with_genes = self.append_multiindex_level(
            intensities.indexes[IntensityIndices.FEATURES.value],
            genes.astype('U'),
            'gene')
        # with_qualities = self.append_multiindex_level(with_genes, qualities, 'quality')

        return IntensityTable(
            intensities=intensities,
            dims=(IntensityIndices.FEATURES.value, Indices.CH.value, Indices.HYB.value),
            coords=(
                with_genes,
                intensities.indexes[Indices.CH.value],
                intensities.indexes[Indices.HYB.value]
            )
        )
