# videoflow-contrib :: tracktor — Tracktor tracker (PyTorch) — GPU.
#
# GPU worker image for the new distributed videoflow (Python 3.12 + CUDA 12.4). Builds
# on videoflow-base:py3.12-cuda and installs CUDA PyTorch wheels. Schedule the pods
# onto GPU nodes (nvidia.com/gpu + the NVIDIA runtime).
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -f tracktor/gpu.Dockerfile -t videoflow-contrib-tracktor:gpu tracktor/
ARG BASE_IMAGE=videoflow-base:py3.12-cuda
FROM ${BASE_IMAGE}

WORKDIR /app

# CUDA 12.4 PyTorch wheels (match the base image's CUDA). torchvision must match torch.
RUN uv pip install --system --break-system-packages --no-cache \
        --index-url https://download.pytorch.org/whl/cu124 \
        'torch>=2.4' 'torchvision>=0.19'

# The videoflow_contrib.tracktor package. --no-deps: videoflow is already in the base.
COPY . ./
RUN uv pip install --system --break-system-packages --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
