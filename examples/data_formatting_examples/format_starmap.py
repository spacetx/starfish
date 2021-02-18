"""
.. _format_starmap:

Format STARmap data
===================

The raw STARmap data received by SpaceTx is organized as z-stack TIFF files that contain all
z-sections for each (round, channel) pair. This TileFetcher breaks these up into individual 2-d
TIFF files, reformats the CSV codebook, and adds the necessary SpaceTx metadata.

This is a good example of converting:

* 3D TIFFS (z, y, x)
* parsing a codebook.csv and writing the SpaceTx Format codebook.json
* working with multiple sets of auxiliary images

.. note::

   This example is provided for illustrative purposes, demonstrating how the
   :py:class:`.TileFetcher` is used in practice. It will need to be adapted to meet
   the specific needs of your data.

The data consists of one field of view with 6 rounds of imaging. Each round is stored in a
multipage TIFF indexed by z. There is also accompanying dapi and nissl-stained images, which are
converted to auxiliary images in SpaceTx Format.

input data structure:
::

    └── parent
        ├── genes.csv
        ├── reg3d
            ├── reg_round01.tif
            ├── reg_round01_ch01.tif
            ├── reg_round01_ch02.tif
            ├── reg_round01_ch03.tif
            ├── reg_round01_ch04.tif
            ├── reg_round02.tif
            ├── reg_round02_ch01.tif
            ├── reg_round02_ch02.tif
            ├── reg_round02_ch03.tif
            ├── reg_round02_ch04.tif
            ├── reg_round03.tif
            ├── ...
        └── nissl
            ├── dapi.tif
            ├── dapi_maxproj.tif
            ├── nissl.tif
            ├── ...

The locations of the data files for use with this script can be found in the
docstring for ``cli``.
"""

import functools
import os
from typing import Mapping, Tuple, Union

import click
import numpy as np
from skimage.io import imread
from slicedimage import ImageFormat

from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, Features


# We use this to cache images. This allows us to write a separate function to extract the data's
# shape from the file, instead of hard-coding the shape. It also enables us to call the function
# once per z-slice, but only read the 3d volume one time.
@functools.lru_cache(maxsize=1)
def cached_read_fn(file_path) -> np.ndarray:
    return imread(file_path)


class StarMapTile(FetchedTile):

    def __init__(
            self,
            file_path: str,
            z: int
    ) -> None:
        """Parser for an osmFISH tile.

        Parameters
        ----------
        file_path : str
            location of the StarMap 3d-TIFF
        coordinates :
            the coordinates for the selected StarMap tile, extracted from the metadata
        z : int
            the z-layer to extract from the StarMap tile
        """
        self.file_path = file_path
        self.z = z

        # dummy coordinates
        self._coordinates = {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: 3738, Axes.X: 17247}  # hard coded for these datasets.

    @property
    def coordinates(self):
        return self._coordinates  # noqa

    def tile_data(self) -> np.ndarray:
        return cached_read_fn(self.file_path)[self.z]  # slice out the correct z-plane


class StarMapTileFetcher(TileFetcher):

    def __init__(self, input_dir: str) -> None:
        """Implement a TileFetcher for a StarMap experiment.

        This TileFetcher constructs spaceTx format for a stitched tissue slice, where
        `input_dir` is a directory containing .tif files with the following structure:

        ├── genes.csv
        └── reg3d
            ├── reg_round01.tif
            ├── reg_round01_ch01.tif
            ├── reg_round01_ch02.tif
            ├── reg_round01_ch03.tif
            ├── reg_round01_ch04.tif
            ├── reg_round02.tif
            ├── reg_round02_ch01.tif
            ├── reg_round02_ch02.tif
            ├── reg_round02_ch03.tif
            ├── reg_round02_ch04.tif
            ├── reg_round03.tif
            ├── reg_round03_ch01.tif
            ├── reg_round03_ch02.tif
            ├── reg_round03_ch03.tif
            ├── reg_round03_ch04.tif
            ├── reg_round04.tif
            ├── reg_round04_ch01.tif
            ├── reg_round04_ch02.tif
            ├── reg_round04_ch03.tif
            ├── reg_round04_ch04.tif
            ├── reg_round05.tif
            ├── reg_round05_ch01.tif
            ├── reg_round05_ch02.tif
            ├── reg_round05_ch03.tif
            ├── reg_round05_ch04.tif
            ├── reg_round06.tif
            ├── reg_round06_ch01.tif
            ├── reg_round06_ch02.tif
            ├── reg_round06_ch03.tif
            └── reg_round06_ch04.tif

        Notes
        -----
        - The spatial organization of the fields of view are not known to the starfish developers,
          so they are filled by dummy coordinates
        - This TileFetcher is specifically tailored to the gene panel used for a specific
          experiment. Generalization of this TileFetcher will require reimplementation of the
          `channel_map` method.
        """

        self.input_dir = input_dir
        self.num_z = 28  # hard coded for this dataset

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        basename = f"reg_round0{round_label+ 1}_ch0{ch_label + 1}.tif"  # translate to 3d
        file_path = os.path.join(self.input_dir, "reg3d", basename)
        return StarMapTile(file_path, zplane_label)

    def generate_codebook(self, output_dir: str) -> None:
        """Generate and save a codebook from the provided mapping of genes to DNA sequences.

        StarMAP codebooks are encoded with the 2-base encoding used for solid sequencing. In this
        scheme, multiple pairs of bases map into the same fluorescence channels. This function
        exposes this mapping.

        Parameters
        ----------
        output_dir : str
            directory in which to save the generated codebook. Codebook is saved as "codebook.json"

        """
        dinucleotides_to_channels = {
            "AT": 4,
            "CT": 3,
            "GT": 2,
            "TT": 1,
            "AG": 3,
            "CG": 4,
            "GG": 1,
            "TG": 2,
            "AC": 2,
            "CC": 1,
            "GC": 4,
            "TC": 3,
            "AA": 1,
            "CA": 2,
            "GA": 3,
            "TA": 4,
        }

        with open(os.path.join(self.input_dir, "genes.csv"), "r") as f:
            codes = [
                line.strip().split(",") for line in f.readlines()
            ]  # List[(gene, dna_barcode), ...]

        def iter_dinucleotides(sequence):
            i = 0
            while i + 1 < len(sequence):
                yield sequence[i:i + 2]
                i += 1

        # construct codebook target mappings
        code_array = []
        for gene, dna_barcode in codes:
            dna_barcode = dna_barcode[::-1]  # reverse barcode
            spacetx_barcode = [
                {
                    Axes.ROUND.value: r,
                    Axes.CH.value: dinucleotides_to_channels[dinucleotide],
                    Features.CODE_VALUE: 1
                } for r, dinucleotide in enumerate(iter_dinucleotides(dna_barcode))
            ]
            code_array.append({
                Features.CODEWORD: spacetx_barcode,
                Features.TARGET: gene
            })

        codebook = Codebook.from_code_array(code_array)
        codebook.to_json(os.path.join(output_dir, "codebook.json"))


class StarMapDapiTileFetcher(TileFetcher):

    def __init__(self, input_dir: str) -> None:
        """Implement a TileFetcher for dapi auxiliary images of a StarMap experiment.


        └── nissl
            ├── dapi.tif
            ├── dapi_maxproj.tif

        """
        self.input_dir = input_dir

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        basename = f"reg_round0{round_label + 1}_ch0{ch_label + 1}.tif"
        file_path = os.path.join(self.input_dir, "reg3d", basename)
        return StarMapTile(file_path, zplane_label)


class StarMapNisslTileFetcher(TileFetcher):

    def __init__(self, input_dir: str, channels: Tuple[int, int]) -> None:
        """Implement a TileFetcher for nissl auxiliary images of a StarMap experiment.

        Images are expected to be found with the following names. Since the nissl images
        correspond to projections across pairs of channels, a separate TileFetcher will be
        initialized for each projection.

        input_dir
        └── nissl
            ├── 2base_BF2_nissl_decon_ch00.tif
            ├── 2base_BF2_nissl_decon_ch01.tif
            ├── 2base_BF2_nissl_decon_ch02.tif
            ├── nissl.tif
            └── nissl_maxproj.tif

        Parameters
        ----------
        input_dir : str
            directory containing input data
        channels : Tuple[int, int]
            the pair of channels corresponding to the TIFF file to be read.

        """
        self.input_dir = input_dir

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        basename = f"reg_round0{round_label + 1}_ch0{ch_label + 1}.tif"
        file_path = os.path.join(self.input_dir, "reg3d", basename)
        return StarMapTile(file_path, zplane_label)


@click.command()
@click.option("--input-dir", type=str, required=True, help="input directory containing images")
@click.option("--output-dir", type=str, required=True, help="output directory for formatted data")
def cli(input_dir, output_dir) -> None:
    """CLI entrypoint for spaceTx format construction for osmFISH data

    Raw data (input for this tool) for this experiment can be found at:
    s3://spacetx.starfish.data.public/browse/raw/20181031/starmap/

    Processed data (output of this tool) can be found at:
    s3://spacetx.starfish.data.public/browse/formatted/20190111/starmap/

    Parameters
    ----------
    input_dir : str
        directory containing input data. See TileFetcher classes for expected directory structures.
    output_dir : str
        directory that 2-d images and SpaceTx metadata will be written to.
    """
    abs_output_dir = os.path.expanduser(output_dir)
    abs_input_dir = os.path.expanduser(input_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    primary_tile_fetcher = StarMapTileFetcher(abs_input_dir)
    dapi_tile_fetcher = StarMapDapiTileFetcher(abs_input_dir)
    nissl_00_tile_fetcher = StarMapNisslTileFetcher(abs_input_dir, channels=(0, 0))
    nissl_01_tile_fetcher = StarMapNisslTileFetcher(abs_input_dir, channels=(0, 1))
    nissl_02_tile_fetcher = StarMapNisslTileFetcher(abs_input_dir, channels=(0, 2))

    # This is hardcoded for this example data set
    primary_image_dimensions: Mapping[Union[str, Axes], int] = {
        Axes.ROUND: 6,
        Axes.CH: 4,
        Axes.ZPLANE: 28,
    }

    aux_images_dimensions: Mapping[str, Mapping[Union[str, Axes], int]] = {
        "nuclei": {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 1,
        },
        "nissl_channels_0_0": {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 1,
        },
        "nissl_channels_0_1": {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 1,
        },
        "nissl_channels_0_2": {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 1,
        },
    }

    write_experiment_json(
        path=output_dir,
        fov_count=1,
        tile_format=ImageFormat.TIFF,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions=aux_images_dimensions,
        primary_tile_fetcher=primary_tile_fetcher,
        aux_tile_fetcher={
            "nuclei": dapi_tile_fetcher,
            "nissl_channels_0_0": nissl_00_tile_fetcher,
            "nissl_channels_0_1": nissl_01_tile_fetcher,
            "nissl_hcannels_0_2": nissl_02_tile_fetcher
        },
        dimension_order=(Axes.ROUND, Axes.CH, Axes.ZPLANE)
    )

    primary_tile_fetcher.generate_codebook(abs_output_dir)


if __name__ == "__main__":
    cli()
