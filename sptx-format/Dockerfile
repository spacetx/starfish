## Dockerfile for sptx
##
## Default entrypoint is the validate-sptx script.

FROM continuumio/miniconda3
RUN useradd -m sptx
USER sptx

# Set up the initial conda environment
COPY --chown=sptx:sptx environment.yml /src/environment.yml
WORKDIR /src
RUN conda env create -f environment.yml

# Prepare for build
COPY --chown=sptx:sptx . /src
RUN echo "source activate sptx" >> ~/.bashrc
ENV PATH /home/sptx/.conda/envs/sptx/bin:$PATH

# Build and configure for running
RUN pip install -e .

env MPLBACKEND Agg
ENTRYPOINT ["validate-sptx"]
