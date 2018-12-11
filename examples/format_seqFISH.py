import functools
import json
import os
from typing import Mapping, Tuple, Union

import click
import numpy as np
import pandas as pd
import skimage.io
from slicedimage import ImageFormat

from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Coordinates, Features, Indices, Number


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
            coordinates: Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]],
            z: int,
            ch: int,
    ):
        self._file_path = file_path
        self._z = z
        self._ch = ch
        self._coordinates = coordinates

    @property
    def shape(self) -> Tuple[int, ...]:
        """Gets image shape directly from the data"""
        return self.tile_data().shape

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        """Stores coordinate information passed from the TileFetcher"""
        return self._coordinates

    @property
    def format(self) -> ImageFormat:
        """Image Format for SeqFISH data is TIFF"""
        return ImageFormat.TIFF

    def tile_data(self) -> np.ndarray:
        """vary z the slowest, then channel -- each round has its own TIFF"""
        return cached_read_fn(self._file_path)[self._z, self._ch]


class SeqFISHTileFetcher(TileFetcher):

    def __init__(self, input_dir: str) -> None:
        """Implement a TileFetcher for a single SeqFISH Field of View."""
        self.input_dir = input_dir

    @property
    def coordinates(self) -> Mapping[Coordinates, Tuple[float, float]]:
        """Returns dummy coordinates for this single-FoV TileFetcher"""
        return {
            Coordinates.X: (0., 0.),
            Coordinates.Y: (0., 0.),
            Coordinates.Z: (0., 0.),
        }

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> SeqFISHTile:
        """Extracts 2-d data from a multi-page TIFF containing all Tiles for an imaging round

        Parameters
        ----------
        fov : int
            Not used in this implementation, will always receive 0
        r : int
            Imaging round. Selects the TIFF file to examine
        ch : int
            Selects the channel from within the loaded TIFF file
        z : int
            Selects the z-plane from within the loaded TIFF file

        Returns
        -------
        SeqFISHTile :
            SeqFISH subclass of FetchedTile
        """
        file_path = os.path.join(self.input_dir, f"{r + 1}.tif")
        return SeqFISHTile(file_path, self.coordinates, z, ch)


def parse_codebook(codebook_csv: str) -> Mapping:
    """Parses a codebook csv file provided by SeqFISH developers.

    Parameters
    ----------
    codebook_csv : str
        The codebook file is expected to contain a matrix whose rows are barcodes and whose columns
        are imaging rounds. Column IDs are expected to be sequential, and round identifiers (roman
        numerals) are replaced by integer IDs.

    Returns
    -------
    Mapping :
        Dictionary representation of a Codebook object that adheres to SpaceTx format.
    """
    csv: pd.DataFrame = pd.read_csv(codebook_csv, index_col=0)
    integer_round_ids = range(csv.shape[1])
    csv.columns = integer_round_ids

    mappings = []

    for gene, channel_series in csv.iterrows():
        mappings.append({
            Features.CODEWORD: [{
                Indices.ROUND.value: r, Indices.CH.value: c - 1, Features.CODE_VALUE: 1
            } for r, c in channel_series.items()],
            Features.TARGET: gene
        })

    return {
        "version": "0.0.0",
        "mappings": mappings
    }


@click.command()
@click.option("--input-dir", type=str, required=True, help="input directory containing images")
@click.option("--output-dir", type=str, required=True, help="output directory for formatted data")
@click.option("--codebook-csv", type=str, required=True, help="csv file containing barcode:target mapping")
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
      s3://spacetx.starfish.data.public/seqfish/
    - Processed data (output of this tool) can be found at:
      s3://spacetx.starfish.data.public/20181211/seqfish/ and accessed in `starfish.data.SeqFISH`

    Returns
    -------
    int :
        Returns 0 if successful
    """
    os.makedirs(output_dir, exist_ok=True)
    primary_tile_fetcher = SeqFISHTileFetcher(os.path.expanduser(input_dir))

    # This is hardcoded for this example data set
    primary_image_dimensions = {
        Indices.ROUND: 5,
        Indices.CH: 12,
        Indices.Z: 29,
    }

    # def post_process_func(experiment_json_doc):
    #     experiment_json_doc["codebook"] = "codebook.json"
    #     return experiment_json_doc

    write_experiment_json(
        path=output_dir,
        fov_count=1,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions={},
        primary_tile_fetcher=primary_tile_fetcher,
        tile_format=ImageFormat.TIFF,
        # postprocess_func=post_process_func,
        dimension_order=(Indices.ROUND, Indices.CH, Indices.Z)
    )

    with open(os.path.join(output_dir, "codebook.json"), "w") as f:
        codebook = parse_codebook(codebook_csv)
        json.dump(codebook, f)

    return 0


if __name__ == "__main__":
    cli()
