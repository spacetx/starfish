"""
.. _format_osmfish:

Format osmFISH Data
===================

The following script formats osmFISH data from primary human visual cortex in SpaceTx Format.
This is a good example of:

* converting 3D .NPY files
* multiple fields of view (FOV)
* not hard coding tile shape (get shape from data)
* reading metadata from YAML file to
  * assign physical coordinates to each FOV
  * get z-tile depth
  * get files named by target gene
  * generate the correct *codebook*

.. note::

   This example is provided for illustrative purposes, demonstrating how the
   :py:class:`.TileFetcher` is used in practice. It will need to be adapted to meet
   the specific needs of your data.

The experiment had hundreds of fields of view but this example selects only 3 to convert.
Each FOV consists of only primary images with 13 rounds and 3 channels. Each NPY is a 3D array (
z, y, x) representing one gene (channel) in one FOV.

input data structure:
::

    └── parent
        ├── Experimental_metadata.yaml
        ├── Hybridization1_Aldoc_fov_53.npy
        ├── Hybridization1_Aldoc_fov_75.npy
        ├── Hybridization1_Aldoc_fov_106.npy
        ├── Hybridization1_Foxj1_fov_53.npy
        ├── Hybridization1_Foxj1_fov_75.npy
        ├── Hybridization1_Foxj1_fov_106.npy
        ├── ...

The locations of the data files for use with this script can be found in the
docstring for ``cli``.
"""

import functools
import json
import os
import re
from typing import Mapping, Union

import click
import numpy as np
from slicedimage import ImageFormat

import starfish.core.util.try_import
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, CoordinateValue, Features


# We use this to cache images across tiles.  In the case of the osmFISH data set, volumes are saved
# together in a single file.  To avoid reopening and decoding the NPY file, we use a single-element
# cache that maps between file_path and the npy file.
@functools.lru_cache(maxsize=1)
def cached_read_fn(file_path) -> np.ndarray:
    return np.load(file_path)


class osmFISHTile(FetchedTile):

    def __init__(
            self,
            file_path: str,
            coordinates: Mapping[Union[str, Coordinates], CoordinateValue],
            z: int
    ) -> None:
        """Parser for an osmFISH tile.

        Parameters
        ----------
        file_path : str
            location of the osmFISH tile
        coordinates : Mapping[Union[str, Coordinates], CoordinateValue]
            the coordinates for the selected osmFISH tile, extracted from the metadata
        z : int
            the z-layer for the selected osmFISH tile
        """
        self.file_path = file_path
        self.z = z
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
        return self._coordinates

    def tile_data(self) -> np.ndarray:
        return cached_read_fn(self.file_path)[self.z]  # slice out the correct z-plane


class osmFISHTileFetcher(TileFetcher):

    @starfish.core.util.try_import.try_import({"yaml"})
    def __init__(self, input_dir: str, metadata_yaml) -> None:
        """Implement a TileFetcher for an osmFISH experiment.

        This TileFetcher constructs spaceTx format for one or more fields of view, where
        `input_dir` is a directory containing all .npy image files and whose file names have the
        following structure:

        Hybridization<round>_<target>_fov_<fov_number>.npy

        Notes
        -----
        - osmFISH is a linearly multiplexed method. As such, each target is specified by a
          (channel, round) tuple. The files do not contain channel information.
        - This TileFetcher is specifically tailored to the gene panel used for a specific
          experiment. Generalization of this TileFetcher will require reimplementation of the
          `channel_map` method.
        """
        import yaml

        with open(metadata_yaml, "r") as f:
            self.osmfish_metadata = yaml.load(f)
        self.num_z = self.osmfish_metadata['ImageProperties']['HybImageSize']['zcount']
        self.input_dir = input_dir

    @property
    def channel_map(self) -> Mapping[str, int]:
        return {
            "Quasar570": 0,
            "CF610": 1,
            "Quasar670": 2,
        }

    @property
    def target_map(self) -> Mapping:
        """
        Parse the metadata to map channel number and round to the correct target, which identifies
        the right tile
        """
        parsed_metadata = {}
        for round_, round_data in self.osmfish_metadata['HybridizationsInfos'].items():

            # the metadata references a larger corpus of data than we use for this example so as we
            # iterate over the metadata, some rounds will not be found. In those cases, we just
            # continue through the loop without adding to parsed_metadata
            round_match = re.match(r"Hybridization(\d{1,2})", round_)
            if round_match is None:
                continue

            round_id = int(round_match.group(1)) - 1
            for target_name, fluorophore in round_data.items():
                if fluorophore in {"Dapi", "FITC"}:
                    continue
                channel = self.channel_map[fluorophore]
                parsed_metadata[round_id, channel] = target_name

        return parsed_metadata

    @property
    def fov_map(self) -> Mapping[int, str]:
        """This example is pared down to only 3 fovs, which are mapped to sequential integers"""
        return {
            0: "53",
            1: "75",
            2: "106",
        }

    def coordinate_map(self, round_: int, z: int):
        pixel_size = self.osmfish_metadata['ImageProperties']['PixelSize']
        y_pixels, x_pixels, z_pixels = (
            self.osmfish_metadata['ImageProperties']['HybImageSize'].values()
        )
        y_size = y_pixels * pixel_size
        x_size = x_pixels * pixel_size
        z_size = z_pixels * pixel_size
        position_string = (
            self.osmfish_metadata['TilesPositions'][f'Hybridization{round_ + 1}'][z]
        )
        y_pos, x_pos, z_pos = (float(v) for v in position_string.split(", "))
        return {
            Coordinates.X: (x_pos, x_pos + x_size),
            Coordinates.Y: (y_pos, y_pos + y_size),
            Coordinates.Z: (z_pos, z_pos + z_size),
        }

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        target = self.target_map[round_label, ch_label]
        fov = self.fov_map[fov_id]
        basename = f"Hybridization{round_label + 1}_{target}_fov_{fov}.npy"
        file_path = os.path.join(self.input_dir, basename)
        coordinates = self.coordinate_map(round_label, zplane_label)
        return osmFISHTile(file_path, coordinates, zplane_label)

    def generate_codebook(self):
        mappings = []
        for (round_, channel), target in self.target_map.items():
            mappings.append({
                Features.CODEWORD: [{
                    Axes.ROUND.value: round_, Axes.CH.value: channel, Features.CODE_VALUE: 1
                }],
                Features.TARGET: target
            })
        return {
            "version": "0.0.0",
            "mappings": mappings
        }


@click.command()
@click.argument("input-dir", type=str)
@click.argument("metadata-yaml", type=str)
@click.argument("output-dir", type=str)
def cli(input_dir, metadata_yaml, output_dir):
    """Reads osmFISH images from <input-dir> and experiment metadata from <metadata-yaml> and writes
    spaceTx-formatted data to <output-dir>.

    Raw data (input for this tool) for this experiment can be found at:
    s3://spacetx.starfish.data.upload/simone/

    Processed data (output of this tool) can be found at:
    s3://spacetx.starfish.data.public/20181031/osmFISH/
    """
    os.makedirs(output_dir, exist_ok=True)
    primary_tile_fetcher = osmFISHTileFetcher(os.path.expanduser(input_dir), metadata_yaml)

    # This is hardcoded for this example data set
    primary_image_dimensions = {
        Axes.ROUND: 13,
        Axes.CH: len(primary_tile_fetcher.channel_map),
        Axes.ZPLANE: primary_tile_fetcher.num_z
    }

    def postprocess_func(experiment_json_doc):
        experiment_json_doc["codebook"] = "codebook.json"
        return experiment_json_doc

    with open(os.path.join(output_dir, "codebook.json"), "w") as f:
        codebook = primary_tile_fetcher.generate_codebook()
        json.dump(codebook, f)

    write_experiment_json(
        path=output_dir,
        fov_count=len(primary_tile_fetcher.fov_map),
        tile_format=ImageFormat.TIFF,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions={},
        primary_tile_fetcher=primary_tile_fetcher,
        postprocess_func=postprocess_func,
        dimension_order=(Axes.ROUND, Axes.CH, Axes.ZPLANE)
    )
    pass


if __name__ == "__main__":
    cli()
