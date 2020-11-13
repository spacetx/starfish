"""
.. _format_seqfish:

Format SeqFISH Data
===================

The following script formats SeqFISH data in SpaceTx Format.
This is a good example of:

* converting 4D TIFFS (channel, z, y, x)
* not hard coding tile shape (get shape from data)
* parsing a codebook.csv and writing the SpaceTx Format codebook.json

.. note::

   This example is provided for illustrative purposes, demonstrating how the
   :py:class:`.TileFetcher` is used in practice. It will need to be adapted to meet
   the specific needs of your data.

The data consists of one field of view with 5 rounds of imaging. Each round is stored in a
multipage TIFF indexed by channel and z.

input data structure:
::

    └── parent
        ├── gene-barcodes.csv
        ├── 1.tif
        ├── 2.tif
        ├── 3.tif
        ├── 4.tif
        ├── 5.tif
        ├── ...

The locations of the data files for use with this script can be found in the
docstring for ``cli``.
"""
import functools
import os
from typing import Mapping, Union

import click
import numpy as np
import pandas as pd
import skimage.io
from slicedimage import ImageFormat

from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, CoordinateValue, Features


# We use this to cache images across tiles.  In the case of the osmFISH data set, volumes are saved
# together in a single file.  To avoid reopening and decoding the TIFF file, we use a single-element
# cache that maps between file_path and the npy file.
@functools.lru_cache(maxsize=1)
def cached_read_fn(file_path) -> np.ndarray:
    return skimage.io.imread(file_path)


class SeqFISHTile(FetchedTile):
    def __init__(
            self,
            file_path: str,
            coordinates: Mapping[Union[str, Coordinates], CoordinateValue],
            zplane: int,
            ch: int,
    ):
        self._file_path = file_path
        self._zplane = zplane
        self._ch = ch
        self._coordinates = coordinates

    @property
    def shape(self) -> Mapping[Axes, int]:
        """
        Gets image shape directly from the data. Note that this will result in the data being
        read twice, since the shape is retrieved from all tiles before the data is read, and thus
        single-file caching does not resolve the duplicated reads.

        Because the data here isn't tremendously large, this is acceptable in this instance.
        """
        raw_shape = self.tile_data().shape
        return {Axes.Y: raw_shape[0], Axes.X: raw_shape[1]}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        """Stores coordinate information passed from the TileFetcher"""
        return self._coordinates

    @property
    def format(self) -> ImageFormat:
        """Image Format for SeqFISH data is TIFF"""
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        """vary z the slowest, then channel -- each round has its own TIFF"""
        return cached_read_fn(self._file_path)[self._zplane, self._ch]


class SeqFISHTileFetcher(TileFetcher):

    def __init__(self, input_dir: str) -> None:
        """Implement a TileFetcher for a single SeqFISH Field of View."""
        self.input_dir = input_dir

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        """Returns dummy coordinates for this single-FoV TileFetcher"""
        return {
            Coordinates.X: (0., 1.),
            Coordinates.Y: (0., 1.),
            Coordinates.Z: (0., 0.1),
        }

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> SeqFISHTile:
        """Extracts 2-d data from a multi-page TIFF containing all Tiles for an imaging round

        Parameters
        ----------
        fov : int
            Not used in this implementation, will always receive 0
        r : int
            Imaging round. Selects the TIFF file to examine
        ch : int
            Selects the channel from within the loaded TIFF file
        zplane : int
            Selects the z-plane from within the loaded TIFF file

        Returns
        -------
        SeqFISHTile :
            SeqFISH subclass of FetchedTile
        """
        file_path = os.path.join(self.input_dir, f"{round_label + 1}.tif")
        return SeqFISHTile(file_path, self.coordinates, zplane_label, ch_label)


def parse_codebook(codebook_csv: str) -> Codebook:
    """Parses a codebook csv file provided by SeqFISH developers.

    Parameters
    ----------
    codebook_csv : str
        The codebook file is expected to contain a matrix whose rows are barcodes and whose columns
        are imaging rounds. Column IDs are expected to be sequential, and round identifiers (roman
        numerals) are replaced by integer IDs.

    Returns
    -------
    Codebook :
        Codebook object in SpaceTx format.
    """
    csv: pd.DataFrame = pd.read_csv(codebook_csv, index_col=0)
    integer_round_ids = range(csv.shape[1])
    csv.columns = integer_round_ids

    mappings = []

    for gene, channel_series in csv.iterrows():
        mappings.append({
            Features.CODEWORD: [{
                Axes.ROUND.value: r, Axes.CH.value: c - 1, Features.CODE_VALUE: 1
            } for r, c in channel_series.items()],
            Features.TARGET: gene
        })

    return Codebook.from_code_array(mappings)


@click.command()
@click.option("--input-dir", type=str, required=True, help="input directory containing images")
@click.option("--output-dir", type=str, required=True, help="output directory for formatted data")
@click.option("--codebook-csv", type=str, required=True,
              help="csv file containing barcode:target mapping")
def cli(input_dir: str, output_dir: str, codebook_csv: str) -> int:
    """CLI entrypoint for spaceTx format construction for SeqFISH data

    Parameters
    ----------
    input_dir : str
        directory containing input multi-page TIFF files for a single field of view, separated by
        the imaging round they were acquired in and named <1-index round>.tif
    output_dir : str
        directory containing output files. Will be created if it does not exist.
    codebook_csv : str
        name of the codebook csv file containing barcode information for this field of view.

    Notes
    -----
    - each round is organized as [z, ch, [x|y], [x|y]] -- the order of x and y are not known, but
      since this script uses dummy coordinates, this distinction is not important.
    - The spatial organization of the field of view is not known to the starfish developers,
      so they are filled by dummy coordinates
    - Raw data (input for this tool) for this experiment can be found at:
      s3://spacetx.starfish.data.public/browse/raw/seqfish/
    - Processed data (output of this tool) can be found at:
      s3://spacetx.starfish.data.public/browse/formatted/20181211/seqfish/ and accessed in
      `starfish.data.SeqFISH`

    Returns
    -------
    int :
        Returns 0 if successful
    """
    os.makedirs(output_dir, exist_ok=True)
    primary_tile_fetcher = SeqFISHTileFetcher(os.path.expanduser(input_dir))

    # This is hardcoded for this example data set
    primary_image_dimensions: Mapping[Union[str, Axes], int] = {
        Axes.ROUND: 5,
        Axes.CH: 12,
        Axes.ZPLANE: 29,
    }

    write_experiment_json(
        path=output_dir,
        fov_count=1,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions={},
        primary_tile_fetcher=primary_tile_fetcher,
        tile_format=ImageFormat.TIFF,
        dimension_order=(Axes.ROUND, Axes.CH, Axes.ZPLANE)
    )

    # Note: this must trigger AFTER write_experiment_json, as it will clobber the codebook with
    # a placeholder.
    codebook = parse_codebook(codebook_csv)
    codebook.to_json("codebook.json")

    return 0


if __name__ == "__main__":
    cli()
