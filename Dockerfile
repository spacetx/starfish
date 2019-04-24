FROM python:3.7.3-stretch

RUN pip install starfish

env MPLBACKEND Agg
