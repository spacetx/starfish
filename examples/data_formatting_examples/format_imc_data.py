"""
.. _format_imc:

Format Imaging Cytof Data
=========================

The following script converts Imaging Cytof Data in SpaceTx Format.

This is a good example of:

* generating codebook from an ordered list of targets
* handling filenames based on target name rather than organized by rounds or channels
* multiple fields of view (FOV)

.. note::

   This example is provided for illustrative purposes, demonstrating how the
   :py:class:`.TileFetcher` is used in practice. It will need to be adapted to meet
   the specific needs of your data.

input data structure:
::

    └── parent
        ├── <Fov1_name>
            └── <Fov1_name>
                ├── <target_name1>.tiff
                ├── ...
        ├── <Fov2_name>
            └── <Fov2_name>
                ├── <target_name1>.tiff
                ├── ...

The locations of the data files for use with this script can be found in ``cli``.
"""
import json
import os
from typing import List, Mapping, Union

import click
import numpy as np
from skimage.io import imread
from slicedimage import ImageFormat

from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, CoordinateValue, Features


class ImagingMassCytometryTile(FetchedTile):

    def __init__(self, file_path: str) -> None:
        """Initialize a TileFetcher for Imaging Mass Cytometry Data"""
        self.file_path = file_path
        self._tile_data = imread(self.file_path)

    @property
    def shape(self) -> Mapping[Axes, int]:
        return {Axes.Y: self._tile_data.shape[0], Axes.X: self._tile_data.shape[1]}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        # dummy coordinates
        return {
            Coordinates.X: (0.0, 0.0001),
            Coordinates.Y: (0.0, 0.0001),
            Coordinates.Z: (0.0, 0.0001),
        }

    def tile_data(self) -> np.ndarray:
        return self._tile_data


class ImagingMassCytometryTileFetcher(TileFetcher):
    def __init__(self, input_dir: str) -> None:
        """Implement a TileFetcher for an Imaging Mass Cytometry Experiment.

        This Tile Fetcher constructs spaceTx format from IMC experiments with a specific directory
        structure:

        input_dir
        └── <Fov_name>
            └── <Fov_name>
                ├── <target_name1>.tiff
                ├── ...
                └── <target_nameN>.tiff

        Notes
        -----
        - In Imaging Mass Cytometry, each channel specifies a unique target, so channel == target
        - Imaging Mass Cytometry experiments have only one imaging round, round is hard coded as 1
        - The spatial organization of the fields of view are not known to the starfish developers,
          so they are filled by dummy coordinates

        """
        self.input_dir = input_dir

    @property
    def _ch_dict(self) -> Mapping[int, str]:
        channels = [
            "CD44(Gd160Di)",
            "CD68(Nd146Di)",
            "CarbonicAnhydraseIX(Er166Di)",
            "Creb(La139Di)",
            "Cytokeratin7(Dy164Di)",
            "Cytokeratin8-18(Yb174Di)",
            "E-cadherin(Er167Di)",
            "EpCAM(Dy161Di)",
            "Fibronectin(Dy163Di)",
            "GATA3(Pr141Di)",
            "Her2(Eu151Di)",
            "HistoneH3(Yb176Di)",
            "Ki-67(Er168Di)",
            "PRAB(Gd158Di)",
            "S6(Er170Di)",
            "SMA(Nd148Di)",
            "Twist(Nd145Di)",
            "Vimentin(Dy162Di)",
            "b-catenin(Ho165Di)",
        ]
        mapping = dict(enumerate(channels))
        return mapping

    def ch_dict(self, ch: int) -> str:
        return self._ch_dict[ch]

    @property
    def _fov_map(self) -> Mapping[int, str]:
        fov_names: List[str] = [
            d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d))
        ]
        mapping = dict(enumerate(fov_names))
        return mapping

    def fov_map(self, fov: int) -> str:
        return self._fov_map[fov]

    def get_tile(
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        fov_name = self.fov_map(fov_id)
        basename = f'{self.ch_dict(ch_label)}.tiff'
        file_path = os.path.join(self.input_dir, fov_name, fov_name, basename)
        return ImagingMassCytometryTile(file_path)

    def generate_codebook(self) -> Mapping:
        mappings = []
        for idx, target in self._ch_dict.items():
            mappings.append({
                Features.CODEWORD: [{
                    Axes.ROUND.value: 0, Axes.CH.value: idx, Features.CODE_VALUE: 1
                }],
                Features.TARGET: target
            })

        return {
            "version": "0.0.0",
            "mappings": mappings
        }


@click.command()
@click.option("--input_dir", type=str, help="input directory containing images")
@click.option("--output_dir", type=str, help="output directory for formatted data")
def cli(input_dir, output_dir):
    """CLI entrypoint for spaceTx format construction for Imaging Mass Cytometry

    Raw data (input for this tool) for this experiment can be found at:
    s3://spacetx.starfish.data.public/browse/raw/20181015/imaging_cytof/\
        BodenmillerBreastCancerSamples/

    Processed data (output of this tool) can be found at:
    s3://spacetx.starfish.data.public/browse/formatted/20181023/imaging_cytof/\
        BodenmillerBreastCancerSamples/

    """

    os.makedirs(output_dir, exist_ok=True)

    primary_tile_fetcher = ImagingMassCytometryTileFetcher(os.path.expanduser(input_dir))
    primary_image_dimensions = {
        Axes.ROUND: 1,
        Axes.CH: len(primary_tile_fetcher._ch_dict),
        Axes.ZPLANE: 1
    }

    def postprocess_func(experiment_json_doc):
        experiment_json_doc["codebook"] = "codebook.json"
        return experiment_json_doc

    with open(os.path.join(output_dir, "codebook.json"), 'w') as f:
        codebook = primary_tile_fetcher.generate_codebook()
        json.dump(codebook, f)

    write_experiment_json(
        path=output_dir,
        fov_count=len(primary_tile_fetcher._fov_map),
        tile_format=ImageFormat.TIFF,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions={},
        primary_tile_fetcher=primary_tile_fetcher,
        postprocess_func=postprocess_func,
    )


if __name__ == "__main__":
    cli()
