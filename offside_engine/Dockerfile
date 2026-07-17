# videoflow-contrib :: offside_engine — stateful Law-11 offside decision engine.
#
# CPU worker image (Python 3.12) building on videoflow-base:py3.12; the worker
# entrypoint (python -m videoflow.worker) is inherited from the base image.
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -t videoflow-contrib-offside_engine:latest offside_engine/
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . ./
RUN uv pip install --system --no-cache .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
