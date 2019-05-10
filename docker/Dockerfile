## Dockerfile for starfish
##
## Default entrypoint is the starfish script,
## but can also be used to run pytests.
##
## Examples:
## --------
##
##   (1) Pull the centrally built image
##   $ docker pull spacetx/starfish:latest
##
##   or, (2) build a local image named "spacetx/starfish"
##   $ docker build -t spacetx/starfish .
##
##   (3) See the help for starfish
##   $ docker run --rm spacetx/starfish -h
##
##   (4) Run starfish passing arguments
##   $ docker run --rm spacetx/starfish [arguments]
##
##   (5) Start bash in the source code directory
##   Useful for development.
##   $ docker run --rm -it --entrypoint=bash spacetx/starfish
##
##   (6) Run the tests on the starfish code base
##   $ docker run --rm -it --entrypoint=pytest spacetx/starfish
##
##   (7) Print the help for pytests
##   $ docker run --rm -it --entrypoint=pytest spacetx/starfish -h
##
##   (8) Run TestWithIssData which downloads test data. The
##   TEST_ISS_KEEP_DATA flag doesn't delete the data so that the
##   data can be extracted from the container with `docker cp`.
##   Since no --rm is passed, the container will need to be
##   cleaned up later.
##
##   $ docker run -e TEST_ISS_KEEP_DATA=true --entrypoint=pytest spacetx/starfish -vsxk TestWithIssData
##
FROM continuumio/miniconda3
RUN useradd -m starfish
USER starfish

# Set up the initial conda environment
COPY --chown=starfish:starfish docker/environment.yml /src/docker/environment.yml
COPY --chown=starfish:starfish docker/pip-config /src/
COPY --chown=starfish:starfish REQUIREMENTS* /src/
WORKDIR /src
ENV PIP_CONFIG_FILE=/src/pip-config
RUN conda env create -f docker/environment.yml \
    && conda clean -tipsy

# Prepare for build
COPY --chown=starfish:starfish . /src
RUN echo "source activate starfish" >> ~/.bashrc
ENV PATH /home/starfish/.conda/envs/starfish/bin:$PATH

# Build and configure for running
RUN pip install -e . --ignore-installed --no-cache-dir

env MPLBACKEND Agg
ENTRYPOINT ["starfish"]
