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

starfish registration -i /tmp/starfish/formatted/hybridization-fov_000.json -o /tmp/starfish/registered/hybridization.json FourierShiftRegistration --reference-stack /tmp/starfish/formatted/nuclei-fov_000.json --upsampling 1000

starfish filter -i /tmp/starfish/registered/hybridization.json -o /tmp/starfish/filtered/hybridization.json WhiteTophat --masking-radius 15
starfish filter -i /tmp/starfish/formatted/nuclei-fov_000.json -o /tmp/starfish/filtered/nuclei.json WhiteTophat --masking-radius 15
starfish filter -i /tmp/starfish/formatted/dots-fov_000.json -o /tmp/starfish/filtered/dots.json WhiteTophat --masking-radius 15

starfish detect_spots --input /tmp/starfish/filtered/hybridization.json --output /tmp/starfish/results GaussianSpotDetector --blobs-stack /tmp/starfish/filtered/dots.json --min-sigma 4 --max-sigma 6 --num-sigma 20 --threshold 0.01

starfish segment --hybridization-stack /tmp/starfish/filtered/hybridization.json --nuclei-stack /tmp/starfish/filtered/nuclei.json -o /tmp/starfish/results/regions.geojson Watershed --dapi-threshold .16 --input-threshold .22 --min-distance 57

starfish target_assignment --coordinates-geojson /tmp/starfish/results/regions.geojson --intensities /tmp/starfish/results/spots.nc --output /tmp/starfish/results/regions.json PointInPoly2D

starfish decode -i /tmp/starfish/results/spots.nc --codebook /tmp/starfish/formatted/codebook.json -o /tmp/starfish/results/spots.nc PerRoundMaxChannelDecoder
```

## interactive visualization in Jupyter notebooks

To use the interactive notebook widgets, run the following commands. Please note that the widget currently 
supports Jupyter notebooks, but not Jupyter Lab. 
```
pip3 install -r REQUIREMENTS-NOTEBOOK.txt
jupyter nbextension enable --py widgetsnbextension
```

## web-based visualization in the browser
This is a work in progess -- based on our output file formats (e.g., geo_json) we have been able to put together a simple prototype in the [starfish-viz](https://github.com/spacetx/starfish-viz) repo.

## development

Clone the repo and look through (CONTRIBUTING.md)[CONTRIBUTING.md]
```
% git clone https://github.com/spacetx/starfish.git
% cd starfish
```

## citing starfish

to cite starfish, please use: 

Axelrod S, Carr AJ, Freeman J, Ganguli D, Long B, Tung T, and others. 
Starfish: Open Source Image Based Transcriptomics and Proteomics Tools, 2018-, 
http://github.com/spacetx/starfish [Online; accessed <date>].

Here’s an example of a BibTeX entry:

@misc{,
author = {Shannon Axelrod, Ambrose J Carr, Jeremy Freeman, Deep Ganguli, Brian Long, Tony Tung, and others},
title = {{Starfish}: Open Source Image Based Transcriptomics and Proteomics Tools},
year = {2018--},
url = "http://github.com/spacetx/starfish",
note = {[Online; accessed <date>]}
}
