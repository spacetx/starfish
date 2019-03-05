import functools
import json
import os
import re
from typing import Mapping, Tuple, Union

import numpy as np
from slicedimage import ImageFormat

import starfish.util.try_import
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, Features, Number
from starfish.util import click


# We use this to cache images across tiles.  In the case of the osmFISH data set, volumes are saved
# together in a single file.  To avoid reopening and decoding the TIFF file, we use a single-element
# cache that maps between file_path and the npy file.
@functools.lru_cache(maxsize=1)
def cached_read_fn(file_path) -> np.ndarray:
    return np.load(file_path)


class osmFISHTile(FetchedTile):

    def __init__(
            self,
            file_path: str,
            coordinates: Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]],
            z: int
    ) -> None:
        """Parser for an osmFISH tile.

        Parameters
        ----------
        file_path : str
            location of the osmFISH tile
        coordinates : Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]
            the coordinates for the selected osmFISH tile, extracted from the metadata
        z : int
            the z-layer for the selected osmFISH tile
        """
        self.file_path = file_path
        self.z = z
        self._coordinates = coordinates

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tile_data().shape

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], Union[Number, Tuple[Number, Number]]]:
        return self._coordinates

    def tile_data(self) -> np.ndarray:
        return cached_read_fn(self.file_path)[self.z]  # slice out the correct z-plane


class osmFISHTileFetcher(TileFetcher):

    @starfish.util.try_import.try_import({"yaml"})
    def __init__(self, input_dir: str, metadata_yaml) -> None:
        """Implement a TileFetcher for an osmFISH experiment.

        This TileFetcher constructs spaceTx format for one or more fields of view, where
        `input_dir` is a directory containing all .npy image files and whose file names have the
        following structure:

        Hybridization<round>_<target>_fov_<fov_number>.npy

        Notes
        -----
        - osmFISH is a non-multiplex method. As such, each target is specified by a
          (channel, round) tuple. The files do not contain channel information,
        - The spatial organization of the fields of view are not known to the starfish developers,
          so they are filled by dummy coordinates
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
        """This example dataset has three channels, which are mapped to sequential integers"""
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

    def get_tile(self, fov: int, r: int, ch: int, z: int) -> FetchedTile:
        target = self.target_map[r, ch]
        fov_id = self.fov_map[fov]
        basename = f"Hybridization{r + 1}_{target}_fov_{fov_id}.npy"
        file_path = os.path.join(self.input_dir, basename)
        coordinates = self.coordinate_map(r, z)
        return osmFISHTile(file_path, coordinates, z)

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
@click.option("--input-dir", type=str, help="input directory containing images")
@click.option("--output-dir", type=str, help="output directory for formatted data")
@click.option("--metadata-yaml", type=str, help="experiment metadata")
def cli(input_dir, output_dir, metadata_yaml):
    """CLI entrypoint for spaceTx format construction for osmFISH data

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
