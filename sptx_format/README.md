# SpaceTx Image Format Specification

## Introduction

This document describes the spaceTx file format specification for image-based biological assays.
The spaceTx format is designed to support the construction of a data tensor, which generalizes the approaches taken by sequential single-molecule FISH and assays that identify targets by building codes over multiple imaging rounds.
Each of theses assays produce images that can form a data tensor.
The data tensor contains a series of (x, y) planar image planes that represent specific z-planes, imaging channels (c), and imaging rounds (r).
Together these form a 5-dimensional tensor (r, c, z, y, x) that serves as a general representation of an image-based transcriptomics or proteomics assay, and is the substrate of the starfish package.

This format should be self-describing and it should specify both how a set of 2-d images form a field of view and how multiple fields of view interact to form a larger experiment.
The spaceTx format accomplishes this by combining these images, stored in 2-dimensional TIFF format, with a series of JSON files that describe how to organize each TIFF file into the 5-dimensional imaging tensor.
Combined with imaging metadata and a pipeline recipe, both of which are defined elsewhere, these files enable a pipeline to generate the desired outputs of a spatial assay: a gene expression matrix augmented with spatial locations of transcripts and cells.

## Format Specification

Here, we tabulate the minimum set of required json files that can describe the imaging data of a spaceTx experiment with brief descriptions of their purpose:

| File name            | Description                                                                                              |
|:---------------------|:---------------------------------------------------------------------------------------------------------|
| experiment.json      | links the data manifests and codebook together                                                           |
| data_manifest.json   | file locations of each field of view                                                       |
| nuclei_manifest.json | file locations for the nuclei images that correspond to each field of view                 |
| field_of_view.json   | describes how individual 2-d image planes form an image tensor                                           |
| codebook.json        | maps patterns of intensity in the channels and rounds of a field of view to target molecules |

Each of these input types and their file formats are described in detail in the following sections.

## Experiment

The data manifest is a JSON file that ties together all information about the data images, auxiliary images (like nuclear stains), and the codebook needed to decode the experiment.
It is the file read by starfish to load data into the analysis environment.

Example:
```json
{
  "version": "0.0.0",
  "images": {
    "primary": "primary_images.json",
    "nuclei": "nuclei.json"
  },
  "codebook": "codebook.json",
  "extras": {
    "is_space_tx_cool": true
  }
}
```

## Manifest

Both the `primary_images.json` and `nuclei.json` files referenced by the above `experiment.json` may contain links to Field of View Manifests (for simple experiments with only one field of view, these fields may also directly reference a field of view).
The Manifest is a simple association of a field of view name with the json file that defines the field of view.
In this example, we demonstrate a primary images manifest with three fields of view.
Such an experiment would likely also have a nuclei manifest, which would _also_ contain three fields of view.

```json
{
  "version": "0.0.0",
  "contents": {
    "fov_000": "primary-images-fov_000.json",
    "fov_001": "primary-images-fov_001.json",
    "fov_00N": "primary-images-fov_002.json"
  },
  "extras": null
}
```

## Field of View

The field of view is the most complex file in the spaceTx format, and must be created for each data tensor and auxiliary image tensor in an experiment.
It provides two key types of information: information about the field of view, and information about each tile contained in it.

The field_of_view.json file specifies the shape of the image tensor, including the size of the (X, Y) image in pixels, and the number of z-planes, imaging channels, and imaging rounds in the experiment.
Thus, an image tensor has shape (r, c, z, y, x), though y and x are limited to at most 3000 pixels.
For experiments that do not leverage all of these concepts, the values can simply be set to one, and that dimension of the tensor will be ignored.
For example, barcoded experiments do not leverage z, and as such, the shape of these experiments will be (r, c, 1, y, x)
In contrast, smFISH experiments may not leverage multiple imaging rounds, but often take optical sections of the tissue through multiple z-planes, and might have shape (1, c, z, y, z).


For each individual tile, the Field of View specifies the portion of the tensor the tile corresponds to by providing the indicies of the tile in (r, c, z), the location of the tile, and the sha256 hash of the file data, to guard against corruption.

Finally, each tile also specifies the coordinates of the image in physical space, relative to some experiment-wide reference point specified in micrometers.

The below example describes a 2-channel, 8-round coded experiment that samples a tissue section using 45 discrete z-planes. For conciseness, the tile data is truncated, and shows only the information for two tiles, while in practice there would be 2 * 8 * 45 tiles.

```json
{
  "version": "0.0.0",
  "dimensions": [
      "x",
      "y",
      "z",
      "r",
      "c"
  ],
  "shape": {
      "z": 45,
      "r": 8,
      "c": 2
  },
  "default_tile_format": "TIFF",
  "default_tile_shape": [
    2048,
    2048
  ],
  "extras": "nothing to see here",
  "tiles": [
    {
      "coordinates": {
        "x": [0.0, 0.1],
        "y": [0.0, 0.1],
        "z": [0.0]
      },
      "indices": {
        "z": 0,
        "r": 0,
        "c": 0
      },
      "file": "fdaj3902.tiff",
      "tile_format": "TIFF",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "extras": {
        "are_we_missing_something":"probably"
      }
    },
    {
      "coordinates": {
        "x": [0.0, 0.1],
        "y": [0.0, 0.1],
        "z": [0.1]
      },
      "indices": {
        "z": 1,
        "r": 1,
        "c": 0
      },
      "file": "F_1_Z_1_H_1_C_0.tiff",
      "tile_format": "TIFF",
 	  "sha256": "e3b0c44i98fx1c149afbf4c8996fb924273941e7649b234ca4959a2b7822b855",
      "extras": {
        "are_we_missing_something": "probably"
      }
    }
  ]
}
```

## Codebook

The final part of the spaceTx specification, the codebook describes how intensities detected in the image tensor correspond to the targets of the assay.
The codebook is an array, where each object in the array lists a codeword and the target it corresponds to.
Each codeword is made up of one or more json objects, each of which describe the expected intensity value for tiles of specific (channel, round) combinations.

For smFISH experiments where each channel corresponds to a different target and there is only one imaging round, the codebook is very simple:

```json
{
  "version": "0.0.0",
  "mappings": [
    {
      "codeword": [
        {"c": 0, "r": 0, "v": 1}
      ],
      "target": "SCUBE2"
    },
    {
      "codeword": [
        {"c": 1, "r": 0, "v": 1}
      ],
      "target": "BRCA"
    },
    {
      "codeword": [
        {"c": 2, "r": 0, "v": 1}
      ],
      "target": "ACTB"
    }
  ]
}
```
In this example, channels 0, 1, and 2 correspond to `SCUBE2`, `BRCA`, and `ACTB`, respectively.
In contrast, a coded experiment may have a more complex codebook:

```json
{
  "version": "0.0.0",
  "mappings": [
    {
      "codeword": [
        {"r": 0, "c": 0, "v": 1},
        {"r": 0, "c": 1, "v": 1}
      ],
      "target": "SCUBE2"
    },
    {
      "codeword": [
        {"r": 0, "c": 0, "v": 1},
        {"r": 1, "c": 1, "v": 1}
      ],
      "target": "BRCA"
    },
    {
      "codeword": [
        {"r": 0, "c": 1, "v": 1},
        {"r": 1, "c": 0, "v": 1}
      ],
      "target": "ACTB"
    }
  ]
}
```

The above example describes the coding scheme of an experiment with 2 rounds and 2 channels, where each code expects exactly two images out of four to produce signal for a given target.
In the above example, a spot in the image tensor would decode to `SCUBE2` if the spot was detected in (round=0, channel=0) and (round=0, channel=1).
