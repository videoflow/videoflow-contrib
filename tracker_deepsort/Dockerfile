# videoflow-contrib :: tracker_deepsort — Deep SORT tracker (NumPy + SciPy).
#
# A deployable worker image for the new distributed videoflow: it builds on
# videoflow-base (framework + broker client + py3.13 interpreter) and adds this
# component's dependencies and package. The worker entrypoint (python -m
# videoflow.worker) is inherited from the base.
#
# Prerequisite: build the base image once (from the videoflow repo root):
#   ./docker/build-images.sh            # produces videoflow-base:py3.13 (and others)
#
# Build this component image (context = this module directory):
#   docker build -t videoflow-contrib-tracker_deepsort:latest tracker_deepsort/
#
# Deploy: pass it as the flow's --image default, or pin it on a node
#   MyNode(name='...', image='videoflow-contrib-tracker_deepsort:latest')
ARG BASE_IMAGE=videoflow-base:py3.13
FROM ${BASE_IMAGE}

WORKDIR /app

# Component dependencies (installed on top of videoflow-base's built-in node deps:
# OpenCV, ffmpeg, NumPy, the NATS client, Redis, PyYAML).
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# The videoflow_contrib.tracker_deepsort package. It must be importable by its module path
# (the path that appears in VF_NODE_CLASS), so install it. --no-deps: videoflow itself
# is already present in the base image.
COPY . ./
RUN uv pip install --system --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from videoflow-base.
