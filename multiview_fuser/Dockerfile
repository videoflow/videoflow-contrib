# videoflow-contrib :: multiview_fuser — multi-view 3D reconstruction / fusion node.
#
# CPU worker image (Python 3.12) building on videoflow-base:py3.12; the worker
# entrypoint (python -m videoflow.worker) is inherited from the base image.
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -t videoflow-contrib-multiview_fuser:latest multiview_fuser/
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . ./
RUN uv pip install --system --no-cache .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
