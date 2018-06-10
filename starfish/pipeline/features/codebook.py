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
    def from_json(cls, json_codebook, num_hybs, num_chs):
        with open(json_codebook, 'r') as f:
            code_array = json.load(f)
        return cls.from_code_array(code_array, num_hybs, num_chs)

    @classmethod
    def from_code_array(cls, code_array, num_hybs, num_chs):

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
            data=np.zeros((len(code_array), num_chs, num_hybs), dtype=np.uint8),
            coords=(
                pd.Index(
                    [d[CodebookIndices.GENE_NAME.value] for d in code_array],
                    name=CodebookIndices.GENE_NAME.value
                ),
                pd.Index(np.arange(4), name=Indices.CH.value),
                pd.Index(np.arange(4), name=Indices.HYB.value),
            )
        )

        # fill the codebook
        for code_dict in code_array:
            codeword = code_dict[CodebookIndices.CODEWORD.value]
            gene = code_dict[CodebookIndices.GENE_NAME.value]
            for letter in codeword:
                code_data.loc[gene, letter[Indices.CH.value], letter[Indices.HYB.value]] = letter[
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

    def decode(self, intensities):
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
