<img src="https://github.com/chanzuckerberg/starfish/raw/master/design/logo.png" width="250">

The goal of *starfish* is to **prototype** a reference pipeline for the analysis of image-based transcriptomics data that works across technologies. This is a **work in progress** and will be developed in the open.

## concept
See this [document](https://docs.google.com/document/d/1IHIngoMKr-Tnft2xOI3Q-5rL3GSX2E3PnJrpsOX5ZWs/edit?usp=sharing) for details. The diagram below describes the core pipeline components and associated file manifests that this package plans to standardize and implement.

![alt text](https://github.com/chanzuckerberg/starfish/raw/master/design/pipeline-diagram.png "candidate pipeline")

## installation
Starfish supports python 3.6 and above. To Install the starfish package, first verify that your python version is compatible. You can check this with pip, which may be called `pip` or `pip3` depending on how you installed python.

The output should look similar to this:
```
% pip3 --version
pip 10.0.1 from /usr/local/lib/python3.6/site-packages/pip (python 3.6)
```

While not required, you may wish to set up a [virtualenv](https://virtualenv.pypa.io/en/stable/). To do this, execute:
```
% python -m venv .venv
```

Install the starfish module in edit-mode and all the dependencies for starfish:
```
% git clone https://github.com/spacetx/starfish.git
% pip install -e starfish
```

## usage
See the [notebook](notebooks/ISS_Simple_tutorial_-_Mouse_vs._Human_Fibroblasts.ipynb) for a fully worked example.

You can also re-produce the notebook results with the command line tool. For usage, run `starfish --help`

Running the commands below will re-produce notebook results.
```
mkdir -p /tmp/starfish/raw
mkdir -p /tmp/starfish/formatted
mkdir -p /tmp/starfish/registered
mkdir -p /tmp/starfish/filtered
mkdir -p /tmp/starfish/results

python examples/get_iss_data.py /tmp/starfish/raw /tmp/starfish/formatted --d 1

starfish register -i /tmp/starfish/formatted/experiment.json -o /tmp/starfish/registered fourier_shift --u 1000

starfish filter -i /tmp/starfish/registered/experiment.json -o /tmp/starfish/filtered/ white_tophat --disk-size 15

starfish show /tmp/starfish/filtered/experiment.json

starfish detect_spots /tmp/starfish/filtered/experiment.json /tmp/starfish/results dots --min_sigma 4 --max_sigma 6  --num_sigma 20 --t 0.01

starfish segment /tmp/starfish/filtered/experiment.json /tmp/starfish/results stain --dt .16 --st .22 --md 57

starfish gene_assignment --coordinates-geojson /tmp/starfish/results/regions.geojson --spots-json /tmp/starfish/results/spots.json --output /tmp/starfish/results/regions.json point_in_poly

starfish decode -i /tmp/starfish/results/encoder_table.json --codebook /tmp/starfish/results/encoder_table.json -o /tmp/starfish/results/decoded_table.json iss
```

## visualization quickstart
To see an interactive web-visualization of the final decoded result, run the following commands

1. [Install nvm](https://github.com/creationix/nvm) if you don't have it
2. Install node `nvm install node`
3. Install budo `npm install budo`
4. From starfish/viz run `npm install`
5. From starfish/viz run `npm start`

To use the interactive notebook widgets, run the following commands. Please note that the widget currently 
supports Jupyter notebooks, but not Jupyter Lab. 
```
pip3 install -r REQUIREMENTS-NOTEBOOK.txt
jupyter nbextension enable --py widgetsnbextension
```

## Development

Clone the repo and look through (CONTRIBUTING.md)[CONTRIBUTING.md]
```
% git clone https://github.com/spacetx/starfish.git
% cd starfish
```

