# videoflow-contrib :: yolo — YOLOv3 detector (TensorFlow-2 + tf-keras legacy Keras 2).
#
# CPU worker image for the new distributed videoflow (Python 3.12). Builds on
# videoflow-base:py3.12 and adds this component's dependencies and package; the worker
# entrypoint (python -m videoflow.worker) is inherited from the base.
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -t videoflow-contrib-yolo:latest yolo/
# A GPU build is available: docker build -f yolo/gpu.Dockerfile -t videoflow-contrib-yolo:gpu yolo/
#
# Deploy: pass it as the flow's --image default, or pin it on a node with image=.
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app
# tf-keras (Keras 2) provides the graph-mode backend this node needs on TF>=2.16.
ENV TF_USE_LEGACY_KERAS=1

# Component dependencies (on top of videoflow-base's built-in node deps: OpenCV,
# ffmpeg, NumPy, the NATS client, Redis, PyYAML).
COPY requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# The videoflow_contrib.yolo package (importable by its module path, which is what
# appears in VF_NODE_CLASS). --no-deps: videoflow is already present in the base image.
COPY . ./
RUN uv pip install --system --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
