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
FROM python:3.6

COPY REQUIREMENTS.txt /src/
COPY REQUIREMENTS-DEV.txt /src/
COPY REQUIREMENTS-NOTEBOOK.txt /src/
RUN pip install -r /src/REQUIREMENTS-DEV.txt -r /src/REQUIREMENTS-NOTEBOOK.txt

RUN useradd -m starfish
COPY . /src
RUN chown -R starfish:starfish /src
USER starfish
WORKDIR /src
RUN pip install --user -e .
ENV PATH=${PATH}:/home/starfish/.local/bin
ENTRYPOINT ["starfish"]
