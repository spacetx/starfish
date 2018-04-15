<img src="https://github.com/chanzuckerberg/starfish/raw/master/design/logo.png" width="250">

The goal of *starfish* is to **prototype** a reference pipeline for the analysis of image-based transcriptomics data that works across technologies. This is a **work in progress** and will be developed in the open. 

## concept
See this [document](https://docs.google.com/document/d/1IHIngoMKr-Tnft2xOI3Q-5rL3GSX2E3PnJrpsOX5ZWs/edit?usp=sharing) for details. The diagram below describes the core pipeline components and associated file manifests that this package plans to standardize and implement.

![alt text](https://github.com/chanzuckerberg/starfish/raw/master/design/pipeline-diagram.png "candidate pipeline")

## usage
Install the starfish package:
```
% pip install starfish
```

See the [notebook](notebooks/Starfish%20Simple%20ISS%20tutorial%20%7C%20Mouse%20vs.%20Human%20Fibroblasts.ipynb) for a fully worked example.

You can also re-produce the notebook results with the command line tool. For usage, run ```starfish --help```

Running the commands below will re-produce notebook results.
```
mkdir -p /tmp/starfish/raw
mkdir -p /tmp/starfish/formatted
mkdir -p /tmp/starfish/registered
mkdir -p /tmp/starfish/filtered
mkdir -p /tmp/starfish/results

python examples/get_iss_data.py /tmp/starfish/raw /tmp/starfish/formatted --d 1

starfish register -i /tmp/starfish/formatted/org.json -o /tmp/starfish/registered fourier_shift --u 1000

starfish filter /tmp/starfish/registered/org.json /tmp/starfish/filtered/ --ds 15

starfish show /tmp/starfish/filtered/org.json

starfish detect_spots /tmp/starfish/filtered/org.json /tmp/starfish/results dots --min_sigma 4 --max_sigma 6  --num_sigma 20 --t 0.01

starfish segment /tmp/starfish/filtered/org.json /tmp/starfish/results stain --dt .16 --st .22 --md 57

starfish decode /tmp/starfish/results --decoder_type iss
```

## visualization quickstart
To see an interactive web-visualization of the final decoded result, run the following commands

1. [Install nvm](https://github.com/creationix/nvm) if you don't have it
2. Install node ```nvm install node```
3. Install budo ```npm install budo```
4. From starfish/viz run ```npm install```
5. From starfish/viz run ```npm start```

## Development

Clone the repo.
```
% git clone git@github.com:chanzuckerberg/starfish.git
% cd starfish
```

### Virtualenv
While not required, you may wish to set up a [virtualenv](https://virtualenv.pypa.io/en/stable/).  If you are using python < 3.3, you should run:

```
% virtualenv .venv
```

If you are using python >= 3.3, you should run:

```
% python -m venv .venv
```

### Installing

Install the starfish module in edit-mode and all the dependencies for starfish:

```
% pip install -e .
% pip install -r REQUIREMENTS.txt
```
