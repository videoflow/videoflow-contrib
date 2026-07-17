# videoflow-contrib :: tracktor — Tracktor tracker (PyTorch) — CPU.
#
# CPU worker image for the new distributed videoflow (Python 3.12). Builds on
# videoflow-base:py3.12 and installs CPU-only PyTorch wheels. The worker entrypoint
# (python -m videoflow.worker) is inherited from the base.
#
# Prerequisite (from the videoflow repo root):  ./docker/build-images.sh
# Build (context = this module directory):
#   docker build -t videoflow-contrib-tracktor:latest tracktor/
# A GPU build is available: docker build -f tracktor/gpu.Dockerfile -t videoflow-contrib-tracktor:gpu tracktor/
#
# Deploy: pass it as the flow's --image default, or pin it on a node with image=.
ARG BASE_IMAGE=videoflow-base:py3.12
FROM ${BASE_IMAGE}

WORKDIR /app

# CPU-only PyTorch wheels (from PyTorch's CPU index). torchvision must match torch.
RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        'torch>=2.2' 'torchvision>=0.17'

# The videoflow_contrib.tracktor package. --no-deps: videoflow is already in the base.
COPY . ./
RUN uv pip install --system --no-cache --no-deps .

# ENTRYPOINT ["python", "-m", "videoflow.worker"] is inherited from the base image.
