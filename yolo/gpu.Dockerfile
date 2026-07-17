# videoflow-contrib :: yolo — YOLOv3 detector (TensorFlow-2 + tf-keras legacy Keras 2) (GPU).
#
# GPU worker image for the new distributed videoflow (Python 3.12 + CUDA). Builds on
# videoflow-base:py3.12-cuda and adds this component's CUDA-enabled dependencies.
# Schedule the resulting pods onto GPU nodes (nvidia.com/gpu + the NVIDIA runtime).
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -f yolo/gpu.Dockerfile -t videoflow-contrib-yolo:gpu yolo/
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app
# tf-keras (Keras 2) provides the graph-mode backend this node needs on TF>=2.16.
ENV TF_USE_LEGACY_KERAS=1

# Component dependencies (on top of videoflow-base's built-in node deps: OpenCV,
# ffmpeg, NumPy, the NATS client, Redis, PyYAML).
COPY requirements-gpu.txt ./requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

# The videoflow_contrib.yolo package (importable by its module path, which is what
# appears in VF_NODE_CLASS). --no-deps: videoflow is already present in the base image.
COPY . ./
RUN uv pip install --system --break-system-packages --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
