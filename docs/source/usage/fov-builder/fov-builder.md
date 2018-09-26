## Building Synthetic SpaceTx-specification compliant experiments

starfish provides a tool to construct example datasets that can be used to test software for use with our formats.
This tool generates spaceTx-specification experiments with tunable sizes and shapes, but the images are randomly generated and do not contain biologically meaningful data.

### Installation

Please follow instructions to install starfish as outlined in the starfish [README](/README.md).

### Usage

starfish build --help will provide instructions on how to use the tool:
```
usage: starfish build [-h] --fov-count FOV_COUNT --hybridization-dimensions
                      HYBRIDIZATION_DIMENSIONS
                      [--dots-dimensions DOTS_DIMENSIONS]
                      [--nuclei-dimensions NUCLEI_DIMENSIONS]
                      output_dir

positional arguments:
  output_dir

optional arguments:
  -h, --help            show this help message and exit
  --fov-count FOV_COUNT
                        Number of FOVs in this experiment.
  --hybridization-dimensions HYBRIDIZATION_DIMENSIONS
                        Dimensions for the hybridization images. Should be a
                        json dict, with r, c, and z as the possible keys. The
                        value should be the shape along that dimension. If a
                        key is not present, the value is assumed to be 0.
  --dots-dimensions DOTS_DIMENSIONS
                        Dimensions for the dots images. Should be a json dict,
                        with r, c, and z as the possible keys. The value
                        should be the shape along that dimension. If a key is
                        not present, the value is assumed to be 0.
  --nuclei-dimensions NUCLEI_DIMENSIONS
                        Dimensions for the nuclei images. Should be a json
                        dict, with r, c, and z as the possible keys. The value
                        should be the shape along that dimension. If a key is
                        not present, the value is assumed to be 0.
```

### Examples:

Build a 3-field of view experiment with 2 channels and 8 hybridization rounds per primary image stack that samples z 30 times.
The experiment has both a dots image and a nuclei image, but these have only one channel and hybridization round each.
The size of the (x,y) tiles cannot be modified at this time.

```bash
mkdir tmp
OUTPUT_DIR=tmp
starfish build \
    --fov-count 3 \
    --hybridization-dimensions '{"r": 8, "c": 2, "z": 30}' \
    --dots-dimensions '{"r": 1, "c": 1, "z": 30}' \
    --nuclei-dimensions '{"r": 1, "c": 1, "z": 30}' \
    ${OUTPUT_DIR}
```
