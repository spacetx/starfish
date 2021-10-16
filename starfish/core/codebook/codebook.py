import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import xarray as xr
from semantic_version import Version
from sklearn.neighbors import NearestNeighbors
from slicedimage.io import resolve_path_or_url

from starfish.core.codebook._format import (
    CURRENT_VERSION,
    DocumentKeys,
    MAX_SUPPORTED_VERSION,
    MIN_SUPPORTED_VERSION,
)
from starfish.core.config import StarfishConfig
from starfish.core.imagestack import indexing_utils
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.spacetx_format.util import CodebookValidator
from starfish.core.types import Axes, Features, Number

NormalizedFeaturesArgtype = TypeVar("NormalizedFeaturesArgtype", "Codebook", IntensityTable)


class Codebook(xr.DataArray):
    """Codebook for an image-based transcriptomics experiment

    The codebook is a three dimensional tensor with shape :code:`(feature, round, channel)` whose
    values are the expected intensity of features (spots or pixels) that correspond to each target
    (gene or protein) in each of the image tiles of an experiment.

    This class supports the construction of synthetic codebooks for testing, and exposes decode
    methods to assign target identifiers to spots. This codebook provides an in-memory
    representation of the codebook defined in the SpaceTx format.

    The codebook is a subclass of xarray, and exposes the complete public API of that package in
    addition to the methods and constructors listed below.

    Examples
    --------
    Build a codebook using :py:meth:`Codebook.synthetic_one_hot_codebook`::

        >>> from starfish import Codebook
        >>> sd = Codebook.synthetic_one_hot_codebook(n_round=4, n_channel=3, n_codes=2)
        >>> sd.codebook()
        <xarray.Codebook (target: 2, r: 4, c: 3)>
        array([[[1, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]],

               [[1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]]], dtype=uint8)
        Coordinates:
          * target   (target) object 6d7fff11-8905-4421-ab49-4f6d8ecdb4b7 \
    1f5f7087-0618-49fc-a6a5-82fee14360b3
          * r        (r) int64 0 1 2 3
          * c        (c) int64 0 1 2


    """

    __slots__ = ()

    @property
    def code_length(self) -> int:
        """return the length of codes in this codebook"""
        return int(np.dot(*self.shape[1:]))

    @classmethod
    def zeros(cls, code_names: Sequence[str], n_round: int, n_channel: int):
        """
        Create an empty codebook of shape (code_names, n_round, n_channel)

        Parameters
        ----------
        code_names : Sequence[str]
            The targets to be coded.
        n_round : int
            Number of imaging rounds used to build the codes.
        n_channel : int
            Number of channels used to build the codes.

        Examples
        --------
        Build an empty 2-round 3-channel codebook::

            >>> from starfish import Codebook
            >>> Codebook.zeros(['ACTA', 'ACTB'], n_round=2, n_channel=4)
            <xarray.Codebook (target: 2, r: 2, c: 4)>
            array([[[0, 0, 0, 0],
                    [0, 0, 0, 0]],

                   [[0, 0, 0, 0],
                    [0, 0, 0, 0]]], dtype=uint8)
            Coordinates:
              * target   (target) object 'ACTA' 'ACTB'
              * r        (r) int64 0 1
              * c        (c) int64 0 1 2 3

        Returns
        -------
        Codebook :
            codebook whose values are all zero

        """
        data = np.zeros((len(code_names), n_round, n_channel), dtype=np.uint8)
        return cls.from_numpy(code_names, n_round, n_channel, data)

    @classmethod
    def from_numpy(
            cls,
            code_names: Sequence[str],
            n_round: int,
            n_channel: int,
            data: np.ndarray,
    ) -> "Codebook":
        """create a codebook of shape (code_names, n_round, n_channel) from a 3-d numpy array

        Parameters
        ----------
        code_names : Sequence[str]
            the targets to be coded
        n_round : int
            number of imaging rounds used to build the codes
        n_channel : int
            number of channels used to build the codes
        data : np.ndarray
            array of unit8 values with len(code_names) x n_channel x n_round elements

        Examples
        --------
        Build a 3-round 4-channel codebook where :code:`ACTA` is specified by intensity in round 0,
        channel 1, and :code:`ACTB` is coded by fluorescence in channels 0, 1, and 2 of rounds 0,
        1, and 2.
        ::

            >>> import numpy as np
            >>> from starfish import Codebook
            >>> data = np.zeros((2,3,4), dtype=np.uint8)
            >>> data[0, 0, 1] = 1                 # ACTA
            >>> data[[1, 1, 1], [0, 1, 2], [0, 1, 2]] = 1  # ACTB
            >>> Codebook.from_numpy(['ACTA', 'ACTB'], n_channel=4, n_round=3, data=data)
            <xarray.Codebook (target: 2, r: 3, c: 4)>
            array([[[0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]],

                   [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]]], dtype=uint8)
            Coordinates:
              * target   (target) object 'ACTA' 'ACTB'
              * r        (r) int64 0 1 2
              * c        (c) int64 0 1 2 3

        Returns
        -------
        Codebook :
            codebook with filled values

        """
        return cls(
            data=data,
            coords=(
                pd.Index(code_names, name=Features.TARGET),
                pd.Index(np.arange(n_round), name=Axes.ROUND.value),
                pd.Index(np.arange(n_channel), name=Axes.CH.value),
            )
        )

    @classmethod
    def _verify_version(cls, semantic_version_str: str) -> None:
        version = Version(semantic_version_str)
        if not (MIN_SUPPORTED_VERSION <= version <= MAX_SUPPORTED_VERSION):
            raise ValueError(
                f"version {version} not supported.  This version of the starfish library only "
                f"supports codebook formats from {MIN_SUPPORTED_VERSION} to "
                f"{MAX_SUPPORTED_VERSION}")

    @classmethod
    def from_code_array(
            cls, code_array: List[Dict[Union[str, Any], Any]],
            n_round: Optional[int] = None, n_channel: Optional[int] = None) -> "Codebook":
        """
        Construct a codebook from a python list of SpaceTx-Format codewords.

        Note: Loading the SpaceTx-Format codebook with :py:meth:`json.load` will produce a code
        array that can be passed to this constructor.

        Parameters
        ----------
        code_array : List[Dict[str, Any]]
            Array of dictionaries, each containing a codeword and target
        n_round : Optional[int]
            The number of imaging rounds used in the codes. Will be inferred if not provided
        n_channel : Optional[int]
            The number of channels used in the codes. Will be inferred if not provided

        Examples
        --------
        Construct a codebook from some array data in python memory
        ::

            >>> from starfish.types import Axes, Features
            >>> from starfish import Codebook
            >>> codebook = [
            >>>     {
            >>>         Features.CODEWORD: [
            >>>             {Axes.ROUND.value: 0, Axes.CH.value: 3, Features.CODE_VALUE: 1},
            >>>             {Axes.ROUND.value: 1, Axes.CH.value: 3, Features.CODE_VALUE: 1},
            >>>         ],
            >>>         Features.TARGET: "ACTB_human"
            >>>     },
            >>>     {
            >>>         Features.CODEWORD: [
            >>>             {Axes.ROUND.value: 0, Axes.CH.value: 3, Features.CODE_VALUE: 1},
            >>>             {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1},
            >>>         ],
            >>>         Features.TARGET: "ACTB_mouse"
            >>>     },
            >>> ]
            >>> Codebook.from_code_array(codebook)
            <xarray.Codebook (target: 2, r: 2, c: 4)>
            array([[[0, 0, 0, 1],
                    [0, 0, 0, 1]],

                   [[0, 0, 0, 1],
                    [0, 1, 0, 0]]], dtype=uint8)
            Coordinates:
              * target   (target) object 'ACTB_human' 'ACTB_mouse'
              * r        (r) int64 0 1
              * c        (c) int64 0 1 2 3

            Returns
            -------
            Codebook :
                Codebook with shape (targets, channels, imaging_rounds)

        """

        # guess the max round and channel if not provided, otherwise check provided values are valid
        max_round, max_ch = 0, 0

        # verify codebook structure and fields
        for code in code_array:

            if not isinstance(code, dict):
                raise ValueError(f'codebook must be an array of dictionary codes. Found: {code}.')

            # verify all necessary fields are present
            required_fields = {Features.CODEWORD, Features.TARGET}
            missing_fields = required_fields.difference(code)
            if missing_fields:
                raise ValueError(
                    f'Each entry of codebook must contain {required_fields}. Missing fields: '
                    f'{missing_fields}')

            for entry in code[Features.CODEWORD]:
                if not isinstance(entry, dict):
                    raise TypeError("codeword entries should be dictionaries")

                required_codeword_fields = {Axes.ROUND.value, Axes.CH.value, Features.CODE_VALUE}
                missing_codeword_fields = required_codeword_fields.difference(entry)
                if missing_codeword_fields:
                    raise ValueError(
                        f"Each codeword entry must contain {required_codeword_fields}. Missing "
                        f"fields: {missing_codeword_fields}")

        for code in code_array:
            for entry in code[Features.CODEWORD]:
                max_round = max(max_round, entry[Axes.ROUND])
                max_ch = max(max_ch, entry[Axes.CH])

        # set n_channel and n_round if either were not provided
        n_round = n_round if n_round is not None else max_round + 1
        n_channel = n_channel if n_channel is not None else max_ch + 1

        # raise errors if provided n_round or n_channel are out of range
        if max_round + 1 > n_round:
            raise ValueError(
                f'code detected that requires an imaging round value ({max_round + 1}) that is '
                f'greater than provided n_round: {max_round}')
        if max_ch + 1 > n_channel:
            raise ValueError(
                f'code detected that requires a channel value ({max_ch + 1}) that is greater '
                f'than provided n_channel: {n_channel}')

        target_names = [w[Features.TARGET] for w in code_array]

        # fill the codebook
        data = np.zeros((len(target_names), n_round, n_channel), dtype=np.uint8)
        for i, code_dict in enumerate(code_array):
            for bit in code_dict[Features.CODEWORD]:
                ch = int(bit[Axes.CH])
                r = int(bit[Axes.ROUND])
                data[i, r, ch] = int(bit[Features.CODE_VALUE])
        return cls.from_numpy(target_names, n_round, n_channel, data)

    @classmethod
    def open_json(
            cls, json_codebook: str,
            n_round: Optional[int] = None,
            n_channel: Optional[int] = None,
    ) -> "Codebook":
        """
        Load a codebook from a SpaceTx Format json file or a url pointing to such a file.

        Parameters
        ----------
        json_codebook : str
            Path or url to json file containing a spaceTx codebook.
        n_round : Optional[int]
            The number of imaging rounds used in the codes. Will be inferred if not provided.
        n_channel : Optional[int]
            The number of channels used in the codes. Will be inferred if not provided.

        Examples
        --------
        Create a codebook from in-memory data
        ::

            >>> from starfish.types import Axes, Features
            >>> from starfish import Codebook
            >>> import tempfile
            >>> import json
            >>> import os
            >>> dir_ = tempfile.mkdtemp()
            >>> codebook = [
            >>>     {
            >>>         Features.CODEWORD: [
            >>>             {Axes.ROUND.value: 0, Axes.CH.value: 3, Features.CODE_VALUE: 1},
            >>>             {Axes.ROUND.value: 1, Axes.CH.value: 3, Features.CODE_VALUE: 1},
            >>>         ],
            >>>         Features.TARGET: "ACTB_human"
            >>>     },
            >>>     {
            >>>         Features.CODEWORD: [
            >>>             {Axes.ROUND.value: 0, Axes.CH.value: 3, Features.CODE_VALUE: 1},
            >>>             {Axes.ROUND.value: 1, Axes.CH.value: 1, Features.CODE_VALUE: 1},
            >>>         ],
            >>>         Features.TARGET: "ACTB_mouse"
            >>>     },
            >>> ]
            >>> # make a fake file
            >>> json_codebook = os.path.join(dir_, 'codebook.json')
            >>> with open(json_codebook, 'w') as f:
            >>>     json.dump(codebook, f)
            >>> # read codebook from file
            >>> Codebook.open_json(json_codebook)
            <xarray.Codebook (target: 2, r: 2, c: 4)>
            array([[[0, 0, 0, 1],
                    [0, 0, 0, 1]],

                   [[0, 0, 0, 1],
                    [0, 1, 0, 0]]], dtype=uint8)
            Coordinates:
              * target   (target) object 'ACTB_human' 'ACTB_mouse'
              * r        (r) int64 0 1
              * c        (c) int64 0 1 2 3

        Returns
        -------
        Codebook :
            Codebook with shape (targets, channels, imaging_rounds)

        """

        config = StarfishConfig()

        backend, name, _ = resolve_path_or_url(json_codebook, backend_config=config.slicedimage)
        with backend.read_contextmanager(name) as fh:
            codebook_doc = json.load(fh)

            if config.strict:
                codebook_validator = CodebookValidator(codebook_doc)
                if not codebook_validator.validate_object(codebook_doc):
                    raise Exception("validation failed")

        if isinstance(codebook_doc, list):
            raise ValueError(
                "codebook is a list and not an dictionary.  It is highly likely that you are using"
                "a codebook formatted for a previous version of starfish.")

        version_str = codebook_doc[DocumentKeys.VERSION_KEY]
        cls._verify_version(version_str)

        return cls.from_code_array(codebook_doc[DocumentKeys.MAPPINGS_KEY], n_round, n_channel)

    def get_partial(self, indexers: Mapping[Axes, Union[int, slice, Sequence]]):
        """
        Slice the codebook data according to the provided indexing parameters. Used in a composite
        codebook scenario.

        Parameters
        ----------
        indexers : Mapping[Axes, Union[int, Sequence]]
            A dictionary of dim:index where index is the value, values or range to index the
            dimension
        """
        selector = indexing_utils.convert_to_selector(indexers)
        return indexing_utils.index_keep_dimensions(self, indexers=selector)

    def to_json(self, filename: Union[str, Path]) -> None:
        """
        Save a codebook to json using SpaceTx Format.

        Parameters
        ----------
        filename : Union[str, Path]
            The name of the file in which to save the codebook.

        """
        code_array = []
        for target_index in range(self.sizes[Features.TARGET]):
            codeword = []
            target_codeword = self[{Features.TARGET: target_index}]
            nonzero_indices = np.nonzero(target_codeword.values)
            for round_index, ch_index in zip(*nonzero_indices):
                codeword.append(
                    {
                        Axes.ROUND.value: int(self.coords[Axes.ROUND.value][round_index]),
                        Axes.CH.value: int(self.coords[Axes.CH.value][ch_index]),
                        Features.CODE_VALUE: float(target_codeword[round_index, ch_index])
                    })
            target_cell: np.ndarray = self.coords[Features.TARGET][target_index].values
            assert len(target_cell.shape) == 0
            target_name = str(target_cell)
            code_array.append({
                Features.CODEWORD: codeword,
                Features.TARGET: target_name,
            })
        codebook_document = {
            DocumentKeys.VERSION_KEY: str(CURRENT_VERSION),
            DocumentKeys.MAPPINGS_KEY: code_array,
        }

        with open(filename, 'w') as f:
            json.dump(codebook_document, f)

    @staticmethod
    def _normalize_features(
            array: NormalizedFeaturesArgtype,
            norm_order: int,
    ) -> Tuple[NormalizedFeaturesArgtype, np.ndarray]:
        """Unit normalize each feature of array

        Parameters
        ----------
        array : Union[Codebook, IntensityTable]
            codebook or intensity table containing (ch, r) features to normalize
        norm_order : int
            the norm to apply to each feature

        Notes
        -----
        The available norms for this function can be found at the following link:
        :doc:`numpy:reference/generated/numpy.linalg.norm`

        Returns
        -------
        Union[IntensityTable, Codebook] :
            IntensityTable or Codebook containing normalized features
        np.ndarray :
            A 1 dimensional numpy array containing the feature norms

        """
        feature_traces = array.stack(traces=(Axes.CH.value, Axes.ROUND.value))
        norm = np.linalg.norm(feature_traces.values, ord=norm_order, axis=1)
        array = array / norm[:, None, None]

        # if a feature is all zero, the information should be spread across the channel
        n = array.sizes[Axes.CH.value] * array.sizes[Axes.ROUND.value]
        partitioned_intensity = np.linalg.norm(np.full(n, fill_value=1 / n), ord=norm_order) / n
        array.values[np.logical_not(np.isfinite(array.values))] = partitioned_intensity

        return array, norm

    @staticmethod
    def _approximate_nearest_code(
            norm_codes: "Codebook", norm_intensities: xr.DataArray, metric: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """find the nearest code for each feature using the ball_tree approximate NN algorithm

        Parameters
        ----------
        norm_codes : Codebook
            codebook with each code normalized to unit length (sum = 1)
        norm_intensities : IntensityTable
            intensity table with each feature normalized to unit length (sum = 1)
        metric : str
            the sklearn metric string to pass to NearestNeighbors

        Returns
        -------
        np.ndarray : metric_output
            the output of metric applied to each feature closest code
        np.ndarray : targets
            the gene that corresponds to each matched code

        Notes
        -----
        This function does not verify that the intensities have been normalized.

        """
        linear_codes = norm_codes.stack(traces=(Axes.CH.value, Axes.ROUND.value)).values
        linear_features = norm_intensities.stack(
            traces=(Axes.CH.value, Axes.ROUND.value)).values

        # reshape into traces
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=metric).fit(linear_codes)
        metric_output, indices = nn.kneighbors(linear_features)
        gene_ids = np.ravel(norm_codes.indexes[Features.TARGET].values[indices])

        return np.ravel(metric_output), gene_ids

    def _validate_decode_intensity_input_matches_codebook_shape(
            self,
            intensities: IntensityTable,
    ):
        """verify that the shapes of the codebook and intensities match"""

        ch_match = intensities.sizes.get(Axes.CH, 0) == self.sizes.get(Axes.CH, 0)
        round_match = intensities.sizes.get(Axes.ROUND, 0) == self.sizes.get(Axes.ROUND, 0)
        if not (ch_match and round_match):
            raise ValueError(
                'Codebook and Intensities must have same number of channels and rounds. '
                f'IntensityTable shape: {intensities.sizes}, Codebook shape: {self.sizes}'
            )

    def decode_metric(
            self, intensities: IntensityTable, max_distance: Number, min_intensity: Number,
            norm_order: int, metric: str = 'euclidean', return_original_intensities: bool = False
    ) -> DecodedIntensityTable:
        """
        Assigns intensity patterns that have been extracted from an :py:class:`ImageStack` and
        stored in an :py:class:`IntensityTable` by a :py:class:`SpotFinder` to the gene targets that
        they encode.

        This method carries out the assignment by first normalizing both the codes and the
        recovered intensities to be unit magnitude using an L2 norm, and then finds the closest
        code for each feature according to a distance metric (default=euclidean).

        Features greater than :code:`max_distance` from their nearest code, or that have an average
        intensity below :code:`min_intensity` are not assigned to any feature.

        Parameters
        ----------
        intensities : IntensityTable
            features to be decoded
        max_distance : Number
            maximum distance between a feature and its closest code for which the coded target will
            be assigned.
        min_intensity : Number
            minimum intensity for a feature to receive a target annotation
        norm_order : int
            the scipy.linalg norm to apply to normalize codes and intensities
        metric : str
            the sklearn metric string to pass to NearestNeighbors
        return_original_intensities: bool
            If True returns original intensity values in the DecodedIntensityTable instead of
            normalized ones (default=False)

        Notes
        -----
        The available norms for this function can be found at the following link:
        :doc:`numpy:reference/generated/numpy.linalg.norm`

        The available metrics for this function can be found at the following link:
        :doc:`scipy:reference/spatial.distance`

        Returns
        -------
        IntensityTable :
            Intensity table containing normalized intensities, target assignments, distances to
            the nearest code, and the filtering status of each feature.

        """

        self._validate_decode_intensity_input_matches_codebook_shape(intensities)

        # add empty metadata fields and return
        if intensities.sizes[Features.AXIS] == 0:
            return DecodedIntensityTable.from_intensity_table(
                intensities,
                targets=(Features.AXIS, np.empty(0, dtype='U')),
                distances=(Features.AXIS, np.empty(0, dtype=np.float64)),
                passes_threshold=(Features.AXIS, np.empty(0, dtype=bool)))

        # normalize both the intensities and the codebook
        norm_intensities, norms = self._normalize_features(intensities, norm_order=norm_order)
        norm_codes, _ = self._normalize_features(self, norm_order=norm_order)

        metric_outputs, targets = self._approximate_nearest_code(
            norm_codes, norm_intensities, metric=metric)

        # only targets with low distances and high intensities should be retained
        passes_filters = np.logical_and(
            norms >= min_intensity,
            metric_outputs <= max_distance,
            dtype=bool
        )

        return_intensities = intensities if return_original_intensities else norm_intensities
        # norm_intensities is a DataArray, make it back into an IntensityTable
        return DecodedIntensityTable.from_intensity_table(
            return_intensities,
            targets=(Features.AXIS, targets),
            distances=(Features.AXIS, metric_outputs),
            passes_threshold=(Features.AXIS, passes_filters))

    def decode_per_round_max(self, intensities: IntensityTable) -> DecodedIntensityTable:
        """
        Assigns intensity patterns that have been extracted from an :py:class:`ImageStack` and
        stored in an :py:class:`IntensityTable` by a :py:class:`SpotFinder` to the gene targets that
        they encode.

        This method carries out the assignment by identifying the maximum-intensity channel for each
        round, and assigning each spot to a code if the maximum-intensity pattern exists in the
        codebook.

        This method is only compatible with one-hot codebooks, where exactly one channel is expected
        to contain fluorescence in each imaging round. This is a common coding strategy for
        experiments that read out one DNA base with a distinct fluorophore in each imaging round.

        Notes
        -----
        - If no code matches the per-round maximum for a feature, it will be assigned 'nan' instead
          of a target value
        - Numpy's argmax breaks ties by picking the first of the tied values -- this can lead to
          unexpected results in low-precision images where some features with "tied" channels will
          decode, but others will be assigned 'nan'.

        Parameters
        ----------
        intensities : IntensityTable
            features to be decoded

        Returns
        -------
        IntensityTable :
            intensity table containing additional data variables for target assignments

        """

        def _view_row_as_element(array: np.ndarray) -> np.ndarray:
            """view an entire code as a single element

            This view allows vectors (codes) to be compared for equality without need for multiple
            comparisons by casting the data in each code to a structured dtype that registers as
            a single value

            Parameters
            ----------
            array : np.ndarray
                2-dimensional numpy array of shape (n_observations, (n_channel * n_round)) where
                observations may be either features or codes.

            Returns
            -------
            np.ndarray :
                1-dimensional vector of shape n_observations

            """
            nrows, ncols = array.shape
            # See: https://numpy.org/doc/stable/user/basics.rec.html#structured-datatype-creation
            dtype = np.dtype(
                {
                    'names': ['f{}'.format(i) for i in range(ncols)],
                    'formats': ncols * [array.dtype]
                }
            )
            return array.view(dtype)

        self._validate_decode_intensity_input_matches_codebook_shape(intensities)

        # add empty metadata fields and return
        if intensities.sizes[Features.AXIS] == 0:
            return DecodedIntensityTable.from_intensity_table(
                intensities,
                targets=(Features.AXIS, np.empty(0, dtype='U')),
                distances=(Features.AXIS, np.empty(0, dtype=np.float64)),
                passes_threshold=(Features.AXIS, np.empty(0, dtype=bool)))

        intensities_without_nans = intensities.fillna(0)
        max_channels = intensities_without_nans.argmax(Axes.CH.value)
        # this snippet of code finds all the (feature, round) spots that have uniform illumination,
        # and assigns them to a ch number that's one larger than max possible to ensure that such
        # spots decode to `NaN`.
        max_channels_max = intensities_without_nans.reduce(np.amax, Axes.CH.value)
        max_channels_min = intensities_without_nans.reduce(np.amin, Axes.CH.value)
        uniform_illumination_mask = (max_channels_max == max_channels_min).values

        max_channels.values[uniform_illumination_mask] = (  # type: ignore
            intensities.sizes[Axes.CH.value]
        )
        codes = self.argmax(Axes.CH.value)

        # TODO ambrosejcarr, dganguli: explore this quality score further
        # calculate distance scores by evaluating the fraction of signal in each round that is
        # found in the non-maximal channels.
        max_intensities = intensities.max(Axes.CH.value)
        round_intensities = intensities.sum(Axes.CH.value)
        distance: IntensityTable = 1 - (max_intensities / round_intensities).mean(Axes.ROUND.value)

        a = _view_row_as_element(codes.values.reshape(self.shape[0], -1))  # type: ignore
        b = _view_row_as_element(
            max_channels.values.reshape(intensities.shape[0], -1)  # type: ignore
        )

        targets = np.full(intensities.shape[0], fill_value=np.nan, dtype=object)

        # decode the intensities
        for i in np.arange(codes.shape[0]):  # type: ignore
            targets[np.where(a[i] == b)[0]] = codes[Features.TARGET][i]

        # a code passes filters if it decodes successfully
        passes_filters = ~pd.isnull(targets)

        return DecodedIntensityTable.from_intensity_table(
            intensities,
            targets=(Features.AXIS, targets.astype('U')),
            distances=(Features.AXIS, distance.data),
            passes_threshold=(Features.AXIS, passes_filters))

    @classmethod
    def synthetic_one_hot_codebook(
            cls, n_round: int, n_channel: int, n_codes: int, target_names: Optional[Sequence] = None
    ) -> "Codebook":
        """Generate codes where one channel is "on" in each imaging round

        Parameters
        ----------
        n_round : int
            number of imaging rounds per code
        n_channel : int
            number of channels per code
        n_codes : int
            number of codes to generate
        target_names : Optional[List[str]]
            if provided, names for targets in codebook

        Examples
        --------
        Create a Codebook with 2 rounds, 3 channels, and 2 codes
        ::

            >>> from starfish import Codebook
            >>> sd = Codebook.synthetic_one_hot_codebook(n_round=4, n_channel=3, n_codes=2)
            >>> sd.codebook()
            <xarray.Codebook (target: 2, r: 4, c: 3)>
            array([[[1, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]],

                   [[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]]], dtype=uint8)
            Coordinates:
              * target   (target) object 6d7fff11-8905-4421-ab49-4f6d8ecdb4b7 \
        1f5f7087-0618-49fc-a6a5-82fee14360b3
              * r        (r) int64 0 1 2 3
              * c        (c) int64 0 1 2

        Returns
        -------
        List[Dict] :
            list of codewords

        """

        codes: Set = set()
        while len(codes) < n_codes:
            codes.add(tuple([np.random.randint(0, n_channel) for _ in np.arange(n_round)]))

        # construct codewords from code
        codewords = [
            [
                {
                    Axes.ROUND.value: h, Axes.CH.value: c, 'v': 1
                } for h, c in enumerate(code)
            ] for code in codes
        ]

        # make a codebook from codewords
        if target_names is None:
            # use a reverse-sorted list of integers as codewords
            target_names = [uuid.uuid4() for _ in range(n_codes)]
        if n_codes != len(target_names):
            raise ValueError(
                f"n_codes ({n_codes} does not match the number of targets ({len(target_names)})")

        codebook = [{Features.CODEWORD: w, Features.TARGET: g}
                    for w, g in zip(codewords, target_names)]

        return cls.from_code_array(codebook, n_round=n_round, n_channel=n_channel)
