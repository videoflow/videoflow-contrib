# videoflow-contrib :: synced_video_reader — per-camera time-offset videofile reader.
#
# CPU worker image (Python 3.12) building on videoflow-base:py3.12; the worker
# entrypoint (python -m videoflow.worker) is inherited from the base image.
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -t videoflow-contrib-synced_video_reader:latest synced_video_reader/
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . ./
RUN uv pip install --system --no-cache .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
