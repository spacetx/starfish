## Development dockerfile for starfish, primarily
## focused on the running of tests but the executables
## are installed to ~starfish/.local/bin
##
## Examples:
##
##   docker build -t starfish .
##   docker run --rm starfish -h
##   docker run --rm -it --entrypoint=bash starfish
##   docker run -e TEST_ISS_KEEP_DATA=true starfish -vsxk iss
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
ENTRYPOINT ["pytest"]
