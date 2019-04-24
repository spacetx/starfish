FROM python:3.7.3-stretch

#RUN python -m venv .venv
#RUN source .venv/bin/activate
RUN pip install starfish

env MPLBACKEND Agg
