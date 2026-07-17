# videoflow-contrib :: detector_tf — TensorFlow-2 object detector (GPU).
#
# GPU worker image for the new distributed videoflow (Python 3.12 + CUDA). Builds on
# videoflow-base:py3.12-cuda and adds this component's CUDA-enabled dependencies.
# Schedule the resulting pods onto GPU nodes (nvidia.com/gpu + the NVIDIA runtime).
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -f detector_tf/gpu.Dockerfile -t videoflow-contrib-detector_tf:gpu detector_tf/
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app
# Component dependencies (on top of videoflow-base's built-in node deps: OpenCV,
# ffmpeg, NumPy, the NATS client, Redis, PyYAML).
COPY requirements-gpu.txt ./requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

# The videoflow_contrib.detector_tf package (importable by its module path, which is what
# appears in VF_NODE_CLASS). --no-deps: videoflow is already present in the base image.
COPY . ./
RUN uv pip install --system --break-system-packages --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
